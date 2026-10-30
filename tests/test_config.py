import json
from pathlib import Path

import pytest
import torch

from eve import defaults
from eve.cli import build_parser, main
from eve.config import INTENTS, PRESETS, ModelConfig, preset
from eve.runtime import eval_mode, pick_device, seed_everything


def test_presets_are_valid():
    for cfg in PRESETS.values():
        assert cfg.head_dim * cfg.n_heads == cfg.d_model
        assert cfg.n_intents == len(INTENTS) == 4


def test_preset_sizes_match_the_plan():
    assert 29e6 < preset("eve-32m").approx_params() < 34e6
    assert preset("tiny").approx_params() < 1e6


def test_preset_overrides_do_not_change_the_preset():
    cfg = preset("eve-32m", dropout=0.0)
    assert cfg.dropout == 0.0
    assert PRESETS["eve-32m"].dropout == 0.1


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
    assert "tiny" in out and "eve-32m" in out and "MB (fp32)" in out


def test_info_counts_one_presets_parameters_per_part(capsys):
    main(["info", "--preset", "tiny"])
    out = capsys.readouterr().out
    assert "embeddings" in out and "total" in out and "MB (fp32)" in out


def with_dropout() -> torch.nn.Module:
    """A small module with parameters (``eval_mode`` reads the device from them) and dropout."""
    return torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))


def test_eval_mode_restores_the_mode_after_an_error() -> None:
    model = with_dropout()
    with pytest.raises(RuntimeError, match="loss failed"), eval_mode(model):
        assert not model.training
        raise RuntimeError("loss failed")
    assert model.training


def test_eval_mode_leaves_an_eval_model_in_eval_mode() -> None:
    model = with_dropout().eval()
    with eval_mode(model):
        assert not model.training
    assert not model.training


def test_compare_shows_rates_and_accuracies_as_percentages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run-a"
    run.mkdir()
    record = {"kind": "val", "step": 10, "val_call_stop_rate": 0.0312, "val_lm": 3.21}
    (run / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    main(["compare", str(run), "--metric", "val_call_stop_rate", "val_lm"])
    out = capsys.readouterr().out
    assert "3.1%" in out and "3.2100" in out


def test_data_build_sizes_default_to_the_shared_defaults() -> None:
    args = build_parser().parse_args(["data", "build"])
    assert (args.templates, args.heldout) == (
        defaults.TEMPLATE_EXAMPLES,
        defaults.HELDOUT_EXAMPLES,
    )
    assert build_parser().parse_args(["data", "build", "--templates", "5k"]).templates == 5000
