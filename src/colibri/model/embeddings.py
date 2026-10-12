import torch
from torch import nn

from colibri.config import ModelConfig


class Embeddings(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.enc_pos = nn.Embedding(cfg.max_encoder_len, cfg.d_model)
        self.dec_pos = nn.Embedding(cfg.max_decoder_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def _embed(self, ids: torch.Tensor, pos: nn.Embedding) -> torch.Tensor:
        n = ids.shape[1]
        if n > pos.num_embeddings:
            raise ValueError(f"sequence of {n} tokens, but only {pos.num_embeddings} positions")
        positions = torch.arange(n, device=ids.device)
        return self.drop(self.tok(ids) + pos(positions))  # (B, n, d) + (n, d) broadcasts

    def encoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d) encoder input vectors."""
        return self._embed(ids, self.enc_pos)

    def decoder(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, n) token ids -> (B, n, d) decoder input vectors."""
        return self._embed(ids, self.dec_pos)
