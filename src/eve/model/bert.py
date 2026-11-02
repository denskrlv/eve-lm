"""A BERT encoder, so the retriever (M11) can start from MiniLM (``all-MiniLM-L6-v2``) instead of
Eve's own pretrained encoder.

MiniLM is a 6-layer BERT (d 384, 12 heads, feed-forward 1,536) distilled from a larger one and
then trained by sentence-transformers on ~1B sentence pairs to put paraphrases close together:
"20 plus 10" near "Add two numbers", which Eve's encoder, pretrained on 900M tokens of span
corruption, doesn't do. Its blocks differ from Eve's ``EncoderBlock`` in three ways, each kept
here so its weights mean what they meant:

- post-LN: each sublayer's output is added to ``x`` and the sum is normalized, ``x =
  LN(x + f(x))``; Eve normalizes the sublayer's input instead, ``x = x + f(LN(x))``;
- exact GELU (with erf), not the tanh approximation;
- LayerNorm eps 1e-12.

Attention is Eve's ``MultiHeadAttention``: BERT's query, key, value and output layers have the
same shapes and biases. ``import_minilm`` reads the sentence-transformers download (safetensors,
read here without the library) into a ``Retriever``, and ``import_minilm_eve`` into an Eve with
MiniLM as its encoder, so training and evaluation use them like any other checkpoint.
"""

import json
import math
import struct
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from eve.config import ModelConfig
from eve.model.attention import MultiHeadAttention
from eve.model.blocks import LayerNorm
from eve.tokenizer import PairTokenizer
from eve.tokenizer.bpe import ARG, BOS, EOS, TOOL, TOOLS, UTT
from eve.tokenizer.wordpiece import WordPieceTokenizer

if TYPE_CHECKING:
    from eve.model.eve import Eve
    from eve.model.retriever import Retriever

EPS = 1e-12  # BERT's LayerNorm eps
SAFETENSORS_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,  # BERT saves its position ids too
}
# Eve's specials sit in BERT's [unusedN] slots, whose embeddings were never trained. <bos> and
# <eos> take [CLS] and [SEP]'s, the markers the embedding of a word that says what follows.
MARKER_WORDS = {UTT: "request", TOOLS: "tools", TOOL: "tool", ARG: "argument"}


class BertEmbeddings(nn.Module):
    """Token plus learned position embeddings, then LayerNorm and dropout. BERT also adds a
    segment ("token type") embedding; the retriever always reads one segment, so
    ``import_minilm`` adds segment 0's vector into the position table instead."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.enc_vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_encoder_len, cfg.d_model)
        self.norm = LayerNorm(cfg.d_model, eps=EPS)
        self.drop = nn.Dropout(cfg.dropout)

    def encoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d), as ``Embeddings.encoder``."""
        n = ids.shape[1]
        if n > self.pos.num_embeddings:
            raise ValueError(
                f"sequence of {n} tokens, but only {self.pos.num_embeddings} positions"
            )
        x = self.tok(ids) + self.pos(torch.arange(n, device=ids.device))
        out: torch.Tensor = self.drop(self.norm(x))
        return out


class BertBlock(nn.Module):
    """A post-LN encoder block: attention, add, norm; feed-forward (exact GELU), add, norm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.attn = MultiHeadAttention(d, cfg.enc_heads, dropout=cfg.dropout)
        self.norm1 = LayerNorm(d, eps=EPS)
        self.ff_in = nn.Linear(d, cfg.ffn_mult * d)
        self.ff_out = nn.Linear(cfg.ffn_mult * d, d)
        self.norm2 = LayerNorm(d, eps=EPS)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` (B, n, d); ``mask`` (B, n) is True for real tokens. Returns (B, n, d)."""
        x = self.norm1(x + self.drop(self.attn(x, key_mask=mask)))
        out: torch.Tensor = self.norm2(x + self.drop(self.ff_out(F.gelu(self.ff_in(x)))))
        return out


def read_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """The tensors of a ``.safetensors`` file: 8 bytes giving the header's length, a JSON header
    of each tensor's dtype, shape and byte range, then the raw data."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    data = memoryview(raw)[8 + n :]
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in SAFETENSORS_DTYPES:
            raise ValueError(f"{path}: {name} has dtype {info['dtype']}")
        start, end = info["data_offsets"]
        flat = torch.frombuffer(bytearray(data[start:end]), dtype=SAFETENSORS_DTYPES[info["dtype"]])
        tensor = flat.reshape(info["shape"])
        out[name] = tensor.float() if tensor.is_floating_point() else tensor
    return out


def minilm_config(bert: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """A ``RetrieverConfig`` (as a dict) of a BERT ``config.json``; ``fields`` add to it."""
    if bert.get("hidden_act") != "gelu" or bert.get("position_embedding_type") != "absolute":
        raise ValueError("only BERT with GELU and absolute positions is supported")
    d = bert["hidden_size"]
    if bert["intermediate_size"] % d:
        raise ValueError(f"feed-forward size {bert['intermediate_size']} isn't a multiple of {d}")
    return {
        "encoder": "bert",
        "vocab_size": bert["vocab_size"],
        "max_encoder_len": bert["max_position_embeddings"],
        "d_model": d,
        "n_heads": bert["num_attention_heads"],
        "n_encoder_layers": bert["num_hidden_layers"],
        "ffn_mult": bert["intermediate_size"] // d,
        "dropout": bert["hidden_dropout_prob"],
        "embed_dim": d,  # the projection starts as the identity: MiniLM's own embeddings
    } | fields


def minilm_state(
    weights: dict[str, torch.Tensor], tokenizer: WordPieceTokenizer, n_layers: int
) -> dict[str, torch.Tensor]:
    """A BERT state dict (Hugging Face names) as a bert ``Retriever``'s, without ``project``
    and ``log_scale``. Fills Eve's special-token rows (see ``MARKER_WORDS``)."""
    w = {k.removeprefix("bert."): v for k, v in weights.items()}
    tok = w["embeddings.word_embeddings.weight"].clone()
    tok[BOS] = tok[tokenizer.ids["[CLS]"]]
    tok[EOS] = tok[tokenizer.ids["[SEP]"]]
    for marker, word in MARKER_WORDS.items():
        tok[marker] = tok[tokenizer.ids[word]]
    pos = (
        w["embeddings.position_embeddings.weight"] + w["embeddings.token_type_embeddings.weight"][0]
    )
    state = {
        "embeddings.tok.weight": tok,
        "embeddings.pos.weight": pos,
        "embeddings.norm.scale": w["embeddings.LayerNorm.weight"],
        "embeddings.norm.shift": w["embeddings.LayerNorm.bias"],
    }
    names = {
        "attn.W_query": "attention.self.query",
        "attn.W_key": "attention.self.key",
        "attn.W_value": "attention.self.value",
        "attn.out_proj": "attention.output.dense",
        "ff_in": "intermediate.dense",
        "ff_out": "output.dense",
    }
    norms = {"norm1": "attention.output.LayerNorm", "norm2": "output.LayerNorm"}
    for i in range(n_layers):
        theirs = f"encoder.layer.{i}."
        for ours, name in names.items():
            for kind in ("weight", "bias"):
                state[f"encoder.{i}.{ours}.{kind}"] = w[f"{theirs}{name}.{kind}"]
        for ours, name in norms.items():
            state[f"encoder.{i}.{ours}.scale"] = w[f"{theirs}{name}.weight"]
            state[f"encoder.{i}.{ours}.shift"] = w[f"{theirs}{name}.bias"]
    return state


def read_minilm(
    directory: str | Path,
) -> tuple[dict[str, Any], WordPieceTokenizer, dict[str, torch.Tensor]]:
    """A sentence-transformers download (``config.json``, ``model.safetensors``, ``vocab.txt``):
    its BERT config, its tokenizer and its weights as a bert ``Retriever``'s (``minilm_state``)."""
    directory = Path(directory)
    bert = json.loads((directory / "config.json").read_text())
    if bert.get("model_type") != "bert":
        raise ValueError(f"{directory} isn't a BERT model ({bert.get('model_type')})")
    tokenizer = WordPieceTokenizer.from_bert_vocab(directory / "vocab.txt")
    if tokenizer.vocab_size != bert["vocab_size"]:
        raise ValueError(
            f"vocab.txt has {tokenizer.vocab_size} pieces, config {bert['vocab_size']}"
        )
    weights = read_safetensors(directory / "model.safetensors")
    return bert, tokenizer, minilm_state(weights, tokenizer, bert["num_hidden_layers"])


def import_minilm(directory: str | Path, **fields: Any) -> tuple["Retriever", WordPieceTokenizer]:
    """A bert ``Retriever`` and its tokenizer from a sentence-transformers download
    (``read_minilm``). Its projection is the identity and it pools by the mean, so before any
    training it scores as the original model does. ``fields`` set other config fields
    (``max_query_len``, ``dropout``)."""
    from eve.model.retriever import SCALE, Retriever, RetrieverConfig

    bert, tokenizer, state = read_minilm(directory)
    cfg = RetrieverConfig(**minilm_config(bert, **fields))
    model = Retriever(cfg)
    state["project.weight"] = torch.eye(cfg.embed_dim, cfg.d_model)
    state["log_scale"] = torch.tensor(math.log(SCALE))
    model.load_state_dict(state)  # strict: every weight has a home
    return model, tokenizer


DECODER = ("embeddings.tok.", "embeddings.dec_pos.", "decoder.", "decoder_norm.", "lm_head.")
FRESH = ("questions.", "enc_case.")  # never trained before the fine-tune


def import_minilm_eve(
    directory: str | Path, decoder_from: str | Path, fresh_decoder: bool = False
) -> tuple["Eve", PairTokenizer]:
    """An Eve whose encoder is MiniLM (``read_minilm``) and whose decoder, its embeddings and LM
    head come from the Eve checkpoint ``decoder_from`` (``eve-32m-pretrain``), with its BPE: it
    already writes text, and only its cross-attention has to learn to read the new encoder. The
    question block and the case embedding start new. ``fresh_decoder`` keeps only the
    tokenizer of ``decoder_from``, for an A/B of what its decoder brings.

    Loading is checked as in ``load_pretrained``: only the new parts may be missing."""
    from eve.model.checkpoint import load_checkpoint
    from eve.model.eve import Eve

    bert, wordpiece, encoder_state = read_minilm(directory)
    base = load_checkpoint(decoder_from)
    if not isinstance(base.model, Eve) or base.model.cfg.encoder != "eve":
        raise ValueError(f"{decoder_from} isn't an Eve with its own encoder")
    d, ffn = base.model.cfg.d_model, base.model.cfg.ffn_mult * base.model.cfg.d_model
    if (bert["hidden_size"], bert["intermediate_size"]) != (d, ffn):
        raise ValueError(
            f"MiniLM is d {bert['hidden_size']}, feed-forward {bert['intermediate_size']}; "
            f"{decoder_from} is d {d}, feed-forward {ffn}"
        )
    cfg = replace(
        base.model.cfg,
        encoder="bert",
        encoder_vocab_size=bert["vocab_size"],
        encoder_heads=bert["num_attention_heads"],
        n_encoder_layers=bert["num_hidden_layers"],
        max_encoder_len=bert["max_position_embeddings"],
    )
    model = Eve(cfg)
    state = {
        ("enc_" + k if k.startswith("embeddings.") else k): v for k, v in encoder_state.items()
    }
    if not fresh_decoder:
        state |= {k: v for k, v in base.model.state_dict().items() if k.startswith(DECODER)}
    result = model.load_state_dict(state, strict=False)
    new = FRESH + (DECODER if fresh_decoder else ())
    if missing := [k for k in result.missing_keys if not k.startswith(new)]:
        raise KeyError(f"no weights for {missing}")
    if result.unexpected_keys:
        raise KeyError(f"weights the model doesn't use: {result.unexpected_keys}")
    return model, PairTokenizer(wordpiece, base.bpe)
