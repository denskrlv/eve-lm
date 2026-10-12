import pytest
import torch

from colibri.config import preset
from colibri.model.embeddings import Embeddings


def test_shapes_and_positions():
    cfg = preset("tiny")
    emb = Embeddings(cfg)
    ids = torch.tensor([[7, 7, 7]])
    enc = emb.encoder(ids)
    assert enc.shape == (1, 3, cfg.d_model)
    assert not torch.allclose(enc[0, 0], enc[0, 1])  # same token, different position
    assert emb.decoder(ids).shape == (1, 3, cfg.d_model)
    assert not torch.allclose(enc, emb.decoder(ids))  # separate position tables
    with pytest.raises(ValueError):
        emb.encoder(torch.zeros(1, cfg.max_encoder_len + 1, dtype=torch.long))


def test_parameter_count_matches_config():
    cfg = preset("small")
    n = sum(p.numel() for p in Embeddings(cfg).parameters())
    assert n == (cfg.vocab_size + cfg.max_encoder_len + cfg.max_decoder_len) * cfg.d_model
