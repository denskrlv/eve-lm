import math

import torch
from torch import nn

from colibri.config import ModelConfig
from colibri.model.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """Layer normalization (book ch. 4.2, `LayerNorm`): give each token's vector mean 0 and
    variance 1 across its d features, then let the model rescale it with a learned ``scale``
    and ``shift``. Same result as ``nn.LayerNorm(d)``; written out to show the steps.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # keeps the division safe when a vector is (almost) constant
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # divide by d, as nn.LayerNorm does
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """GELU with the tanh approximation GPT-2 used (book ch. 4.3). Like ReLU for large |x|, but
    smooth around 0 and slightly negative there, so small negative inputs still get a gradient.
    Same result as ``F.gelu(x, approximate="tanh")``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    """The per-token MLP (book ch. 4.3, `FeedForward`): widen to ``ffn_mult · d``, GELU, narrow
    back. Attention mixes information between tokens; this layer works on each token alone.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.layers = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), GELU(), nn.Linear(hidden, cfg.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def attention(cfg: ModelConfig, causal: bool = False, rope: bool = False) -> MultiHeadAttention:
    """A ``MultiHeadAttention`` with the config's dropout and QK-norm; RoPE only if ``rope``
    and the config turns it on."""
    return MultiHeadAttention(
        cfg.d_model,
        cfg.n_heads,
        causal=causal,
        dropout=cfg.dropout,
        rope=rope and cfg.rope,
        rope_base=cfg.rope_base,
        qk_norm=cfg.qk_norm,
    )


class EncoderBlock(nn.Module):
    """The book's `TransformerBlock` (ch. 4.5) with the causal mask off: every token sees every
    other token, before and after it.

    Pre-LN, as in GPT-2: each sublayer reads a normalized copy of ``x`` and adds its output back
    onto ``x`` (the shortcut of ch. 4.4), so the residual stream itself is never normalized
    until the final norm after the last block.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = LayerNorm(cfg.d_model)
        self.attn = attention(cfg, rope=True)
        self.norm2 = LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)  # the book's drop_shortcut

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` (B, n, d); ``mask`` (B, n) is True for real tokens. Returns (B, n, d)."""
        x = x + self.drop(self.attn(self.norm1(x), key_mask=mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """An encoder block with a cross-attention sublayer between self-attention and the
    feed-forward layer. Cross-attention is how the decoder reads the encoder: queries come from
    the decoder, keys and values from the encoder output (``memory``). The book's GPT-2 has no
    encoder, so it has no such sublayer; the original Transformer (Vaswani et al.) does.

    ``causal=True`` is the value decoder: token i sees tokens 0..i of its own sequence.
    ``causal=False`` is the question block: the learned queries all see each other. Its
    queries have no order, so it never uses RoPE; cross-attention never does either.
    """

    def __init__(self, cfg: ModelConfig, causal: bool = True):
        super().__init__()
        d, p = cfg.d_model, cfg.dropout
        self.norm1 = LayerNorm(d)
        self.self_attn = attention(cfg, causal=causal, rope=causal)
        self.norm2 = LayerNorm(d)
        self.cross_attn = attention(cfg)
        self.norm3 = LayerNorm(d)
        self.ff = FeedForward(cfg)
        self.drop = nn.Dropout(p)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x`` (B, n, d) and its ``mask`` (B, n); ``memory`` (B, m, d), the encoder output,
        and its ``memory_mask`` (B, m). Returns (B, n, d)."""
        x = x + self.drop(self.self_attn(self.norm1(x), key_mask=mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), kv=memory, key_mask=memory_mask))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x
