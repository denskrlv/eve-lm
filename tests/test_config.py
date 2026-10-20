import json

import pytest
import torch

from colibri import defaults
from colibri.cli import build_parser, main
from colibri.config import INTENTS, PRESETS, ModelConfig, preset
from colibri.runtime import eval_mode, pick_device, seed_everything


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


def test_cli_dtype_choices_match_the_training_loop():
    from colibri.cli import DTYPE_CHOICES
    from colibri.train.config import DTYPES

    assert DTYPE_CHOICES == list(DTYPES)


def test_eval_mode_restores_the_mode_after_an_error():
    model = torch.nn.Dropout(0.5)
    with pytest.raises(RuntimeError), eval_mode(model):
        assert not model.training
        raise RuntimeError("loss failed")
    assert model.training


def test_compare_shows_rates_and_accuracies_as_percentages(tmp_path, capsys):
    run = tmp_path / "run-a"
    run.mkdir()
    record = {"kind": "val", "step": 10, "val_call_stop_rate": 0.0312, "val_lm": 3.21}
    (run / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    main(["compare", str(run), "--metric", "val_call_stop_rate", "val_lm"])
    out = capsys.readouterr().out
    assert "3.1%" in out and "3.2100" in out


def test_data_build_sizes_default_to_the_shared_defaults():
    args = build_parser().parse_args(["data", "build"])
    assert (args.templates, args.heldout) == (
        defaults.TEMPLATE_EXAMPLES,
        defaults.HELDOUT_EXAMPLES,
    )
    assert build_parser().parse_args(["data", "build", "--templates", "5k"]).templates == 5000
