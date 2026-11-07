from pathlib import Path

import pytest
import torch

from eve.config import preset
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve
from eve.model.splitter import Splitter, SplitterConfig, load_splitter
from eve.tokenizer.bpe import BPETokenizer


def tiny() -> Splitter:
    torch.manual_seed(0)
    return Splitter(SplitterConfig(**preset("tiny").to_dict()))


def test_splitter_shapes() -> None:
    model = tiny()
    ids = torch.randint(50, 100, (2, 16))
    out = model(ids, torch.ones(2, 16, dtype=torch.bool))
    assert out.exists.shape == (2, 8) and out.members.shape == (2, 8, 16)


def test_padding_is_masked() -> None:
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, 10:] = False
    out = tiny()(torch.randint(50, 100, (2, 16)), mask)
    assert torch.isinf(out.members[1, :, 10:]).all() and (out.members[1, :, 10:] < 0).all()
    assert torch.isfinite(out.members[0]).all()


def test_start_from_eve_checkpoint(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    torch.manual_seed(1)
    eve = Eve(preset("tiny"))
    save_checkpoint(tmp_path / "eve.pt", eve, tokenizer, {"task": "questions"})
    ckpt = Splitter.start(tmp_path / "eve.pt")
    for name, tensor in eve.state_dict().items():
        if name.startswith(("embeddings.", "encoder.", "encoder_norm.")):
            assert torch.equal(ckpt.model.state_dict()[name], tensor), name
    assert ckpt.model.cfg.n_actions == 8

    save_checkpoint(tmp_path / "split.pt", ckpt.model, tokenizer, {"task": "split"})
    again = Splitter.start(tmp_path / "split.pt")  # a splitter checkpoint: all of it
    for name, tensor in ckpt.model.state_dict().items():
        assert torch.equal(again.model.state_dict()[name], tensor), name
    assert isinstance(load_checkpoint(tmp_path / "split.pt").model, Splitter)
    assert isinstance(load_splitter(tmp_path / "split.pt").model, Splitter)
    with pytest.raises(ValueError, match="isn't a splitter"):
        load_splitter(tmp_path / "eve.pt")


def test_start_refuses_unknown_overrides(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    save_checkpoint(tmp_path / "eve.pt", Eve(preset("tiny")), tokenizer)
    with pytest.raises(ValueError, match="d_model"):
        Splitter.start(tmp_path / "eve.pt", d_model=32)
