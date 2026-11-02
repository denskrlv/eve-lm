import json
import struct
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from eve.api.toolbox import Dense
from eve.data.schema import Tool
from eve.model.bert import BertBlock, import_minilm, read_safetensors
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.retriever import Retriever, RetrieverConfig, padded
from eve.tokenizer.bpe import BOS, EOS, SPECIAL_TOKENS, TOOL
from eve.tokenizer.wordpiece import WordPieceTokenizer

PIECES = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "request", "tools", "tool", "argument"]
PIECES += ["set", "the", "lights", "lock", "door", "play", "music", "_", ":", "on", "off"]
D, HEADS, LAYERS, FFN = 16, 4, 2, 32
BERT = {
    "model_type": "bert",
    "hidden_act": "gelu",
    "position_embedding_type": "absolute",
    "hidden_size": D,
    "intermediate_size": FFN,
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "max_position_embeddings": 64,
    "hidden_dropout_prob": 0.1,
    "vocab_size": len(SPECIAL_TOKENS) + len(PIECES),
}


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header, chunks, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.contiguous().numpy().tobytes()
        dtype = "I64" if tensor.dtype == torch.int64 else "F32"
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))


def bert_weights() -> dict[str, torch.Tensor]:
    """Random weights under Hugging Face's BERT names."""
    g = torch.Generator().manual_seed(0)

    def r(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * 0.2

    w = {
        "embeddings.position_ids": torch.arange(64)[None],
        "embeddings.word_embeddings.weight": r(BERT["vocab_size"], D),
        "embeddings.position_embeddings.weight": r(64, D),
        "embeddings.token_type_embeddings.weight": r(2, D),
        "embeddings.LayerNorm.weight": 1 + r(D),
        "embeddings.LayerNorm.bias": r(D),
        "pooler.dense.weight": r(D, D),  # unused: the retriever pools by the mean
    }
    for i in range(LAYERS):
        p = f"encoder.layer.{i}."
        for name in ("attention.self.query", "attention.self.key", "attention.self.value"):
            w[p + name + ".weight"], w[p + name + ".bias"] = r(D, D), r(D)
        w[p + "attention.output.dense.weight"], w[p + "attention.output.dense.bias"] = r(D, D), r(D)
        w[p + "intermediate.dense.weight"], w[p + "intermediate.dense.bias"] = r(FFN, D), r(FFN)
        w[p + "output.dense.weight"], w[p + "output.dense.bias"] = r(D, FFN), r(D)
        for norm in ("attention.output.LayerNorm", "output.LayerNorm"):
            w[p + norm + ".weight"], w[p + norm + ".bias"] = 1 + r(D), r(D)
    return w


@pytest.fixture
def download(tmp_path: Path) -> Path:
    """A sentence-transformers download of a tiny random BERT."""
    lines = ["[PAD]", *(f"[unused{i}]" for i in range(len(SPECIAL_TOKENS) - 1)), *PIECES]
    (tmp_path / "vocab.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "config.json").write_text(json.dumps(BERT))
    write_safetensors(tmp_path / "model.safetensors", bert_weights())
    return tmp_path


def test_safetensors_round_trip(tmp_path: Path) -> None:
    tensors = {"a": torch.randn(3, 4), "ids": torch.arange(5)}
    write_safetensors(tmp_path / "x.safetensors", tensors)
    loaded = read_safetensors(tmp_path / "x.safetensors")
    assert torch.equal(loaded["a"], tensors["a"]) and torch.equal(loaded["ids"], tensors["ids"])


def test_bert_block_matches_pytorch_post_ln_layer() -> None:
    cfg = RetrieverConfig(d_model=D, n_heads=HEADS, ffn_mult=FFN // D, dropout=0.0)
    torch.manual_seed(0)
    block = BertBlock(cfg).eval()
    ref = nn.TransformerEncoderLayer(
        D, HEADS, FFN, dropout=0.0, activation="gelu", layer_norm_eps=1e-12, batch_first=True
    ).eval()
    a = block.attn
    with torch.no_grad():
        ref.self_attn.in_proj_weight.copy_(
            torch.cat([a.W_query.weight, a.W_key.weight, a.W_value.weight])
        )
        ref.self_attn.in_proj_bias.copy_(torch.cat([a.W_query.bias, a.W_key.bias, a.W_value.bias]))
        ref.self_attn.out_proj.load_state_dict(a.out_proj.state_dict())
        ref.linear1.load_state_dict(block.ff_in.state_dict())
        ref.linear2.load_state_dict(block.ff_out.state_dict())
        for ours, theirs in ((block.norm1, ref.norm1), (block.norm2, ref.norm2)):
            theirs.weight.copy_(ours.scale)
            theirs.bias.copy_(ours.shift)
    x = torch.randn(2, 5, D)
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    expected = ref(x, src_key_padding_mask=~mask)
    assert torch.allclose(block(x, mask)[0], expected[0], atol=1e-5)
    assert torch.allclose(block(x, mask)[1, :3], expected[1, :3], atol=1e-5)


def test_import_pools_like_sentence_transformers(download: Path) -> None:
    model, tokenizer = import_minilm(download)
    model.eval()
    w = bert_weights()
    ids = [BOS, *tokenizer.encode("lock the door"), EOS]
    x = w["embeddings.word_embeddings.weight"][
        [tokenizer.ids[t] for t in ("[CLS]", "lock", "the", "door", "[SEP]")]
    ]
    x = (
        x
        + w["embeddings.position_embeddings.weight"][:5]
        + w["embeddings.token_type_embeddings.weight"][0]
    )
    x = F.layer_norm(
        x, (D,), w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"], 1e-12
    )
    for i in range(LAYERS):  # BERT by hand
        p = f"encoder.layer.{i}."

        def lin(name: str, h: torch.Tensor, p: str = p) -> torch.Tensor:
            return F.linear(h, w[p + name + ".weight"], w[p + name + ".bias"])

        q, k, v = (
            lin(f"attention.self.{n}", x).view(5, HEADS, -1).transpose(0, 1)
            for n in ("query", "key", "value")
        )
        attn = torch.softmax(q @ k.transpose(-1, -2) / (D // HEADS) ** 0.5, -1) @ v
        h = lin("attention.output.dense", attn.transpose(0, 1).reshape(5, D))
        x = F.layer_norm(
            x + h,
            (D,),
            w[p + "attention.output.LayerNorm.weight"],
            w[p + "attention.output.LayerNorm.bias"],
            1e-12,
        )
        h = lin("output.dense", F.gelu(lin("intermediate.dense", x)))
        x = F.layer_norm(
            x + h, (D,), w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"], 1e-12
        )
    expected = F.normalize(x.mean(0), dim=-1)
    got = model(*padded([ids, ids[:2]]))[0]  # a padded neighbour changes nothing
    assert torch.allclose(got, expected, atol=1e-5)


def test_imported_retriever_saves_trains_and_scores(download: Path, tmp_path: Path) -> None:
    model, tokenizer = import_minilm(download)
    assert model.cfg.arch == "bert" and model.cfg.embed_dim == D
    assert torch.equal(
        model.embeddings.tok.weight[TOOL], model.embeddings.tok.weight[tokenizer.ids["tool"]]
    )
    save_checkpoint(tmp_path / "r.pt", model, tokenizer, {"task": "retrieval"})
    loaded = load_checkpoint(tmp_path / "r.pt")
    assert isinstance(loaded.tokenizer, WordPieceTokenizer)
    assert isinstance(loaded.model, Retriever) and loaded.model.cfg.arch == "bert"
    with pytest.raises(TypeError, match="BPE"):
        _ = loaded.bpe  # an Eve model's paths refuse it
    started = Retriever.start(tmp_path / "r.pt", max_tool_len=32)  # the run's init
    assert started.model.cfg.tool_len == 32

    tools = [Tool("lock_door", "lock the door", ()), Tool("play_music", "play music", ())]
    scores = Dense(loaded.model.eval(), loaded.tokenizer).scores("lock the door", tools)
    assert len(scores) == 2 and all(-1.0 <= s <= 1.0 for s in scores)


def test_unknown_arch_is_refused() -> None:
    with pytest.raises(ValueError, match="arch"):
        Retriever(RetrieverConfig(d_model=D, n_heads=HEADS, arch="gpt"))
