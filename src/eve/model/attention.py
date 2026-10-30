import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019): LayerNorm without the mean and the shift. Each vector is
    divided by its root mean square, then rescaled per feature by a learned ``scale``.

    Used for QK-norm on each head's queries and keys: their dot products, and with them the
    attention logits, can then no longer grow without bound as training goes on.
    """

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide each vector by its root mean square, then apply the scale."""
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def rope_angles(
    n: int, head_dim: int, base: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, each (n, head_dim), of rotary position embeddings (RoPE, Su et al. 2021).

    Feature pair (i, i + head_dim/2) of the vector at position m is rotated by the angle
    m · base^(-2i/head_dim): the first pairs turn fast, the last ones slowly, like the hands
    of a clock.
    """
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.arange(n, device=device).float()[:, None] * inv_freq  # (n, head_dim / 2)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each position's vector by its angles. ``x`` (..., n, head_dim).

    Rotating the query at position m and the key at position n turns their dot product into a
    function of the offset m - n only, so attention sees how far apart two tokens are, not
    where they are. A learned position table has to learn "the token right before me" again
    for every one of its positions.
    """
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)  # (a, b) -> (-b, a) per pair
    return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention, used three ways (book ch. 3.6, `MultiHeadAttention`):

    - bidirectional self-attention: the encoder and the question block (``causal=False``);
    - causal self-attention: the decoder (``causal=True``), as in the book;
    - cross-attention: decoder and question block reading the encoder (``kv`` given).

    Contract the tests check:
    - Attributes ``W_query``, ``W_key``, ``W_value`` and ``out_proj``: each ``nn.Linear(d_model,
      d_model)`` with a bias (4·d² + 4·d parameters, as ``ModelConfig.approx_params`` counts).
    - ``ValueError`` if ``d_model`` isn't divisible by ``n_heads``.
    - ``use_sdpa`` is a plain attribute the tests flip on the same module: False runs the manual
      path, True runs ``F.scaled_dot_product_attention``. Both must give the same output.

    Two options, both off by default:
    - ``rope``: rotary positions on the queries and keys of self-attention. Cross-attention
      never uses them: the decoder's and the encoder's positions don't count in the same frame.
    - ``qk_norm``: ``q_norm`` and ``k_norm``, RMSNorms over head_dim shared by all heads, applied
      to the queries and keys before RoPE (2·head_dim more parameters).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
        rope: bool = False,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout
        self.use_sdpa = False  # see the class docstring; models switch it with set_sdpa
        self.rope = rope
        self.rope_base = rope_base
        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # mixes the heads' outputs, as in the book
        self.drop = nn.Dropout(dropout)
        if qk_norm:
            self.q_norm: RMSNorm | None = RMSNorm(self.head_dim)
            self.k_norm: RMSNorm | None = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend from ``x`` to ``kv`` (to ``x`` itself when ``kv`` is None).

        Shapes: ``x`` (B, n_q, d) gives the queries. ``kv`` (B, n_kv, d) gives keys and values.
        ``key_mask`` (B, n_kv) is True for real tokens and False for padding: the ``enc_mask`` or
        ``dec_mask`` from the collate. Padding keys get zero attention weight.
        Returns (B, n_q, d). With ``return_weights`` (always the manual path), also returns the
        weights (B, n_heads, n_q, n_kv) after softmax, for plotting.

        ``causal=True`` with ``kv`` given is a ``ValueError``: cross-attention is never causal.

        The book's steps: project to Q, K, V -> split into heads -> scores = Q·Kᵀ / √head_dim ->
        mask (future positions if causal, padding keys) with -inf -> softmax -> dropout ->
        weights·V -> merge heads -> out_proj.
        """
        if self.causal and kv is not None:
            raise ValueError("cross-attention is never causal")
        self_attention = kv is None
        kv = x if kv is None else kv  # self-attention: queries, keys and values all come from x
        b, n_q, d = x.shape
        q, k, v = self._project(x, kv, self_attention)  # steps 1 and 1b

        # 2. Which query may look at which key (True = allowed), broadcast to the scores' shape.
        allowed = self._allowed(key_mask, n_q, kv.shape[1], x.device)

        weights: torch.Tensor | None = None
        if self.use_sdpa and not return_weights:
            # PyTorch's fused kernel does steps 3-5 in one call. Its dropout doesn't know about
            # train/eval mode, so it is switched off outside training.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            out, weights = self._manual_attention(q, k, v, allowed)  # steps 3 to 5

        # 6. Merge the heads back into one vector per query, then let out_proj mix them.
        merged: torch.Tensor = self.out_proj(out.transpose(1, 2).reshape(b, n_q, d))
        if return_weights:
            assert weights is not None  # return_weights always takes the manual path
            return merged, weights
        return merged

    def _project(
        self, x: torch.Tensor, kv: torch.Tensor, self_attention: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Steps 1 and 1b: queries from ``x``, keys and values from ``kv``, each split into
        heads, (B, heads, n, head_dim); then the optional norms and rotary positions."""
        # 1. Project, then split d into n_heads × head_dim: (B, n, d) -> (B, heads, n, head_dim).
        #    Each head works on its own head_dim-sized slice, like the book's view + transpose.
        q = self._split(self.W_query(x))
        k = self._split(self.W_key(kv))
        v = self._split(self.W_value(kv))
        # 1b. Options: normalize the queries and keys, then rotate them by their positions.
        if self.q_norm is not None and self.k_norm is not None:  # both or neither
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope and self_attention:
            cos, sin = rope_angles(x.shape[1], self.head_dim, self.rope_base, x.device)
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)  # bf16 under autocast
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 3 to 5, written out: the heads' outputs (B, heads, n_q, head_dim) and the
        attention weights (B, heads, n_q, n_kv)."""
        # 3. Scaled dot-product scores (ch. 3.5): how well each query matches each key.
        #    Dividing by √head_dim keeps the scores from growing with the vector size.
        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5  # (B, heads, n_q, n_kv)
        # 4. Forbidden pairs get -inf, so softmax gives them exactly zero weight.
        if allowed is not None:
            scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # each row sums to 1
        # 5. Each query's output is a weighted average of the values (dropout as in ch. 3.5.2).
        out: torch.Tensor = self.drop(weights) @ v  # (B, heads, n_q, head_dim)
        return out, weights

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

    def _allowed(
        self, key_mask: torch.Tensor | None, n_q: int, n_kv: int, device: torch.device
    ) -> torch.Tensor | None:
        """(B, 1, n_q, n_kv) boolean mask, or None when every query may see every key.

        The book builds the causal mask with ``torch.triu`` and marks *forbidden* positions.
        Here True means *allowed* (the convention of ``scaled_dot_product_attention``), so the
        causal part is the lower triangle: query i sees keys 0..i.
        """
        allowed = None
        if key_mask is not None:
            allowed = key_mask[:, None, None, :].bool()  # (B, 1, 1, n_kv): padding keys off
        if self.causal:
            past = torch.ones(n_q, n_kv, dtype=torch.bool, device=device).tril()
            allowed = past if allowed is None else allowed & past
        return allowed
