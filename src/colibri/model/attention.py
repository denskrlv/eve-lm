import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention, used three ways (book ch. 3.6, `MultiHeadAttention`):

    - bidirectional self-attention: the encoder and the question block (``causal=False``);
    - causal self-attention: the decoder (``causal=True``), as in the book;
    - cross-attention: decoder and question block reading the encoder (``kv`` given).

    Contract the tests check:
    - Attributes ``W_query``, ``W_key``, ``W_value`` and ``out_proj``: each ``nn.Linear(d_model,
      d_model)`` with a bias (4·d² + 4·d parameters, as ``ModelConfig.approx_params`` counts).
    - ``ValueError`` if ``d_model`` isn't divisible by ``n_heads``.
    - ``use_sdpa`` is a plain attribute the tests flip on the same module: False runs your manual
      path, True runs ``F.scaled_dot_product_attention``. Both must give the same output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
        use_sdpa: bool = False,
    ):
        super().__init__()
        raise NotImplementedError

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
        Returns (B, n_q, d). With ``return_weights`` (manual path only), also returns the weights
        (B, n_heads, n_q, n_kv) after softmax, for plotting.

        ``causal=True`` with ``kv`` given is a ``ValueError``: cross-attention is never causal.

        The book's steps: project to Q, K, V -> split into heads -> scores = Q·Kᵀ / √head_dim ->
        mask (future positions if causal, padding keys) with -inf -> softmax -> dropout ->
        weights·V -> merge heads -> out_proj.
        """
        raise NotImplementedError
