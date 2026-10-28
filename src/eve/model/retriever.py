"""The retriever (M11): picks which of many registered tools a request is shown.

A bi-encoder: Eve's encoder reads a request or a tool schema on its own, the mean of its output
vectors goes through a linear layer and is normalized to length 1. A request and a tool match
as well as the cosine of their vectors. Tools are embedded once, when registered; a request
costs one encoder pass, however many tools there are. Eve's own encoder can't do this: it reads
the request together with at most 5 tools.

It starts from a pretrained Eve's embeddings and encoder (``start``) and learns, contrastively,
to put each request close to the tools that serve it (``eve.tasks.retrieval``).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from eve.config import ModelConfig
from eve.data.render import bucket
from eve.data.schema import Tool
from eve.model.attention import MultiHeadAttention
from eve.model.blocks import EncoderBlock, LayerNorm
from eve.model.checkpoint import Checkpoint, _read
from eve.model.embeddings import Embeddings
from eve.model.eve import INIT_STD
from eve.registry import register
from eve.runtime import eval_mode
from eve.tokenizer.bpe import BOS, EOS, PAD, TOOLS, UTT, BPETokenizer

BODY = ("embeddings.", "encoder.", "encoder_norm.")  # the weights a pretrained Eve gives it
SCALE = 20.0  # the starting inverse temperature of the cosines (learned; CLIP starts at 14)
MAX_SCALE = 100.0  # CLIP's cap: a larger scale makes the loss unstable
EMBED_BATCH = 128  # texts per encoder pass when embedding many
LENGTHS = (32, 64, 128, 256, 512)  # sequences are padded to one of these: on MPS, every new
# shape grows the allocator's cache, which it gives back rarely


@dataclass(frozen=True)
class RetrieverConfig(ModelConfig):
    """An Eve config (only the embeddings and encoder fields are used), the size of the
    vectors requests and tools are compared by, and the most tokens of a request and of a tool
    schema it reads. Most are shorter; a long schema loses the end of its last arguments, while
    its name and description, which come first, are always read. Training pads every batch to
    these lengths, so they bound its memory."""

    embed_dim: int = 256
    max_query_len: int = 128
    max_tool_len: int = 128

    @property
    def query_len(self) -> int:
        return min(self.max_query_len, self.max_encoder_len)

    @property
    def tool_len(self) -> int:
        return min(self.max_tool_len, self.max_encoder_len)


def query_ids(tokenizer: BPETokenizer, text: str, max_len: int) -> list[int]:
    """A request as the encoder reads it: as in Eve's prompt, without the tools."""
    ids = tokenizer.encode(" " + text.strip())[: max_len - 3]
    return [BOS, UTT, *ids, EOS]


def tool_ids(tokenizer: BPETokenizer, tool: Tool, max_len: int) -> list[int]:
    """A tool schema as the encoder reads it: as in Eve's prompt, without the request."""
    ids = [i for part in tool.render_parts() for i in tokenizer.encode(part, specials=True)]
    return [BOS, TOOLS, *ids[: max_len - 3], EOS]


def padded(
    seqs: Sequence[Sequence[int]], length: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, n) ids padded with ``PAD`` to ``length`` (default: the longest), and the mask of
    real tokens."""
    n = max(len(s) for s in seqs) if length is None else length
    ids = torch.full((len(seqs), n), PAD, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids, ids != PAD


@register("model", "retriever")
class Retriever(nn.Module):
    """Embeddings -> encoder blocks -> final norm, as in ``Eve``; then the mean over the real
    tokens, a linear projection to ``embed_dim`` and L2 normalization."""

    def __init__(self, cfg: RetrieverConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.n_encoder_layers))
        self.encoder_norm = LayerNorm(cfg.d_model)
        self.project = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(SCALE)))
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding):
                nn.init.normal_(module.weight, std=INIT_STD)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Retriever":
        """A new retriever of the ``RetrieverConfig`` saved as ``config``."""
        return cls(RetrieverConfig(**config))

    @classmethod
    def start(
        cls, path: str | Path, device: str | torch.device = "cpu", **overrides: Any
    ) -> Checkpoint:
        """The retriever to train, from a checkpoint at ``path``:

        - an Eve checkpoint (pretrained or fine-tuned): its embeddings and encoder, with a new
          projection. ``overrides`` may set ``embed_dim``, the lengths and ``dropout``.
        - a retriever checkpoint: all of it, to train on; ``overrides`` may set the lengths
          (``max_query_len``, ``max_tool_len``) and ``dropout``.

        Loading is checked as in ``load_pretrained``: only the new parts may be missing.
        """
        data = _read(path)
        retriever = data.get("model") == "retriever"
        lengths = {"dropout", "max_query_len", "max_tool_len"}  # change no weight's shape
        allowed = lengths | (set() if retriever else {"embed_dim"})
        if bad := set(overrides) - allowed:
            raise ValueError(f"can't change {sorted(bad)} when starting from {path}")
        names = {f.name for f in fields(RetrieverConfig)}
        config = {k: v for k, v in data["config"].items() if k in names} | overrides
        model = cls(RetrieverConfig(**config))
        state = (
            data["state_dict"]
            if retriever
            else {k: v for k, v in data["state_dict"].items() if k.startswith(BODY)}
        )
        result = model.load_state_dict(state, strict=False)
        new = () if retriever else ("project.", "log_scale")
        if missing := [k for k in result.missing_keys if not k.startswith(new)]:
            raise KeyError(f"{path} has no weights for {missing}")
        if result.unexpected_keys:
            raise KeyError(
                f"{path} has weights the retriever doesn't use: {result.unexpected_keys}"
            )
        tokenizer = BPETokenizer.from_dict(data["tokenizer"])
        return Checkpoint(model.to(device), tokenizer, data["meta"])  # type: ignore[arg-type]

    def set_sdpa(self, enabled: bool) -> None:
        """Switch every attention layer to PyTorch's fused kernel (fast) or the manual path."""
        for m in self.modules():
            if isinstance(m, MultiHeadAttention):
                m.use_sdpa = enabled

    @property
    def scale(self) -> torch.Tensor:
        """The inverse temperature the loss multiplies cosines by, capped at ``MAX_SCALE``."""
        return self.log_scale.exp().clamp(max=MAX_SCALE)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids and mask -> (B, embed_dim) vectors of length 1."""
        x = self.embeddings.encoder(ids)
        for block in self.encoder:
            x = block(x, mask)
        x = self.encoder_norm(x)
        weights = mask.unsqueeze(-1).to(x.dtype)
        mean = (x * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return F.normalize(self.project(mean), dim=-1)

    @torch.no_grad()
    def embed(self, seqs: Sequence[Sequence[int]], batch_size: int = EMBED_BATCH) -> torch.Tensor:
        """(len(seqs), embed_dim) vectors on the CPU, in ``seqs``' order, with dropout off.
        Sequences of similar length go through together, so little padding is computed."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = torch.empty(len(seqs), self.cfg.embed_dim)
        with eval_mode(self) as device:
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                longest = max(len(seqs[i]) for i in chunk)
                ids, mask = padded([seqs[i] for i in chunk], bucket(longest, LENGTHS))
                out[chunk] = self(ids.to(device), mask.to(device)).float().cpu()
        return out

    def embed_queries(self, texts: Sequence[str], tokenizer: BPETokenizer) -> torch.Tensor:
        """The vector of each request."""
        return self.embed([query_ids(tokenizer, text, self.cfg.query_len) for text in texts])

    def embed_tools(self, tools: Sequence[Tool], tokenizer: BPETokenizer) -> torch.Tensor:
        """The vector of each tool schema."""
        return self.embed([tool_ids(tokenizer, tool, self.cfg.tool_len) for tool in tools])
