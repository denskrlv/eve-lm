import torch
from torch import nn

from eve.config import ModelConfig


class Embeddings(nn.Module):
    """Token embeddings plus learned position embeddings (book ch. 2.8). With ``cfg.rope`` there
    are no position tables: attention rotates its queries and keys by position instead. Without
    ``encoder`` (an Eve whose encoder has its own, ``eve.model.bert``), only the decoder's."""

    def __init__(self, cfg: ModelConfig, encoder: bool = True):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.max_len = {"encoder": cfg.max_encoder_len, "decoder": cfg.max_decoder_len}
        self.enc_pos: nn.Embedding | None = None
        self.dec_pos: nn.Embedding | None = None
        if not cfg.rope:
            if encoder:
                self.enc_pos = nn.Embedding(cfg.max_encoder_len, cfg.d_model)
            self.dec_pos = nn.Embedding(cfg.max_decoder_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def _embed(self, ids: torch.Tensor, pos: nn.Embedding | None, side: str) -> torch.Tensor:
        n = ids.shape[1]
        if n > self.max_len[side]:
            raise ValueError(f"sequence of {n} tokens, but only {self.max_len[side]} positions")
        x = self.tok(ids)
        if pos is not None:
            x = x + pos(torch.arange(n, device=ids.device))  # (B, n, d) + (n, d) broadcasts
        out: torch.Tensor = self.drop(x)
        return out

    def encoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d) encoder input vectors."""
        return self._embed(ids, self.enc_pos, "encoder")

    def decoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d) decoder input vectors."""
        return self._embed(ids, self.dec_pos, "decoder")
