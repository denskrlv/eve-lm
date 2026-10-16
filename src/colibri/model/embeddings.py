import torch
from torch import nn

from colibri.config import ModelConfig


class Embeddings(nn.Module):
    """Token embeddings plus learned position embeddings (book ch. 2.8). With ``cfg.rope`` there
    are no position tables: attention rotates its queries and keys by position instead."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.max_len = {"encoder": cfg.max_encoder_len, "decoder": cfg.max_decoder_len}
        if cfg.rope:
            self.enc_pos = self.dec_pos = None
        else:
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
        return self.drop(x)

    def encoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d) encoder input vectors."""
        return self._embed(ids, self.enc_pos, "encoder")

    def decoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d) decoder input vectors."""
        return self._embed(ids, self.dec_pos, "decoder")
