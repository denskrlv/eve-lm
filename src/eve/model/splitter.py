"""The splitter: cuts a request into simple actions by pointing at its words.

Eve's encoder reads the request alone (no tools), as the retriever does. A head built like
Eve's question block then asks ``n_actions`` fixed questions, "what is action #i?": learned
queries go through non-causal decoder blocks that cross-attend to the encoder output. Each
answer has two parts:

- ``exists``: one logit, whether action #i is there;
- ``members``: one logit per token, whether that token's word belongs to action #i, as a
  pointer score ``(W_q·h_i)·(W_k·m_t)/√pointer_dim`` plus a per-slot bias, the form of Eve's
  tool pointer. A word can be in several actions (copied) or in none (dropped).

Only each word's first token is labeled (``eve.split.words``); the others are scored but
not trained. It starts from an Eve checkpoint's embeddings and encoder (``start``).
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eve.config import ModelConfig
from eve.model.attention import set_sdpa
from eve.model.blocks import DecoderBlock, EncoderBlock, LayerNorm
from eve.model.checkpoint import Checkpoint, _read, load_checkpoint
from eve.model.embeddings import Embeddings
from eve.model.eve import INIT_STD
from eve.registry import register
from eve.tokenizer.bpe import BPETokenizer

BODY = ("embeddings.", "encoder.", "encoder_norm.")  # the weights an Eve checkpoint gives it
HEAD = ("queries", "layers.", "norm.", "exists_head.", "W_q.", "W_k.", "slot_bias")
LENGTHS = (32, 64, 128)  # requests are padded to one of these (fixed shapes for MPS)


@dataclass(frozen=True)
class SplitterConfig(ModelConfig):
    """An Eve config (only the embeddings and encoder fields are used, and ``pointer_dim``),
    the number of action slots, the head's layers, and the most tokens of a request it reads;
    a longer request is never split."""

    n_actions: int = 8
    n_split_layers: int = 2
    max_split_len: int = 128


@dataclass
class SplitLogits:
    """``exists`` (B, n_actions) and ``members`` (B, n_actions, n) logits; padded tokens'
    members are -inf."""

    exists: torch.Tensor
    members: torch.Tensor


@register("model", "splitter")
class Splitter(nn.Module):
    """Embeddings -> encoder blocks -> final norm, as in ``Eve``; then the action head."""

    def __init__(self, cfg: SplitterConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.n_encoder_layers))
        self.encoder_norm = LayerNorm(cfg.d_model)
        self.queries = nn.Parameter(torch.randn(cfg.n_actions, cfg.d_model) * INIT_STD)
        self.layers = nn.ModuleList(
            DecoderBlock(cfg, causal=False) for _ in range(cfg.n_split_layers)
        )
        self.norm = LayerNorm(cfg.d_model)
        self.exists_head = nn.Linear(cfg.d_model, 1)
        self.W_q = nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.pointer_dim, bias=False)
        self.slot_bias = nn.Parameter(torch.zeros(cfg.n_actions))
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, std=INIT_STD)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Splitter":
        """A new splitter of the ``SplitterConfig`` saved as ``config``."""
        return cls(SplitterConfig(**config))

    @classmethod
    def start(
        cls, path: str | Path, device: str | torch.device = "cpu", **overrides: Any
    ) -> Checkpoint["Splitter"]:
        """The splitter to train, from a checkpoint at ``path``: an Eve checkpoint (its
        embeddings and encoder, with a new head; ``overrides`` may set ``n_actions``,
        ``n_split_layers``, ``max_split_len`` and ``dropout``), or a splitter checkpoint (all
        of it; ``overrides`` may set ``max_split_len`` and ``dropout``). Only the new head
        may be missing from the file."""
        data = _read(path)
        splitter = data.get("model") == "splitter"
        allowed = {"dropout", "max_split_len"} | (
            set() if splitter else {"n_actions", "n_split_layers"}
        )
        if bad := set(overrides) - allowed:
            raise ValueError(f"can't change {sorted(bad)} when starting from {path}")
        names = {f.name for f in fields(SplitterConfig)}
        config = {k: v for k, v in data["config"].items() if k in names} | overrides
        model = cls(SplitterConfig(**config))
        state = (
            data["state_dict"]
            if splitter
            else {k: v for k, v in data["state_dict"].items() if k.startswith(BODY)}
        )
        result = model.load_state_dict(state, strict=False)
        new = () if splitter else HEAD
        if missing := [k for k in result.missing_keys if not k.startswith(new)]:
            raise KeyError(f"{path} has no weights for {missing}")
        if result.unexpected_keys:
            raise KeyError(f"{path} has weights the splitter doesn't use: {result.unexpected_keys}")
        tokenizer = BPETokenizer.from_dict(data["tokenizer"])
        return Checkpoint(model.to(device), tokenizer, data["meta"])

    def set_sdpa(self, enabled: bool) -> None:
        """Switch every attention layer to PyTorch's fused kernel (fast) or the manual path."""
        set_sdpa(self, enabled)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> SplitLogits:
        """(B, n) token ids and mask -> the action head's logits."""
        x = self.embeddings.encoder(ids)
        for block in self.encoder:
            x = block(x, mask)
        memory = self.encoder_norm(x)
        h = self.queries.expand(ids.shape[0], -1, -1)
        for layer in self.layers:
            h = layer(h, memory, memory_mask=mask)
        h = self.norm(h)
        exists = self.exists_head(h).squeeze(-1)
        q, k = self.W_q(h), self.W_k(memory)
        members = q @ k.transpose(1, 2) / q.shape[-1] ** 0.5 + self.slot_bias[:, None]
        members = members.masked_fill(~mask[:, None, :], float("-inf"))
        return SplitLogits(exists, members)


def load_splitter(path: str | Path, device: str | torch.device = "cpu") -> Checkpoint[Splitter]:
    """``load_checkpoint`` for a file that must hold a ``Splitter``."""
    ckpt = load_checkpoint(path, device)
    if not isinstance(model := ckpt.model, Splitter):
        raise ValueError(f"{path} isn't a splitter")
    return Checkpoint(model, ckpt.tokenizer, ckpt.meta, ckpt.train_state)
