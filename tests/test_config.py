import pytest
import torch

from colibri.cli import main
from colibri.config import INTENTS, PRESETS, ModelConfig, preset
from colibri.runtime import pick_device, seed_everything


def test_presets_are_valid():
    for cfg in PRESETS.values():
        assert cfg.head_dim * cfg.n_heads == cfg.d_model
        assert cfg.n_intents == len(INTENTS) == 4


def test_preset_sizes_match_the_plan():
    assert 11e6 < preset("small").approx_params() < 13.5e6
    assert 29e6 < preset("base").approx_params() < 34e6
    assert 50e6 < preset("medium").approx_params() < 60e6
    assert preset("tiny").approx_params() < 1e6


def test_preset_overrides_do_not_change_the_preset():
    cfg = preset("small", dropout=0.0)
    assert cfg.dropout == 0.0
    assert PRESETS["small"].dropout == 0.1


def test_unknown_preset():
    with pytest.raises(KeyError):
        preset("huge")


def test_invalid_heads():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=3)


def test_seeding_is_repeatable():
    seed_everything(0)
    a = torch.randn(4)
    seed_everything(0)
    assert torch.equal(a, torch.randn(4))


def test_pick_device_respects_explicit_name():
    assert pick_device("cpu") == torch.device("cpu")


def test_info_command(capsys):
    main(["info"])
    out = capsys.readouterr().out
    assert "small" in out and "base" in out
