from pathlib import Path

import pytest

from eve import defaults
from eve.cli import build_parser, parse_args
from eve.run_configs import apply_config, available, config_path, read_config

REPO_CONFIGS = Path(__file__).parents[1] / defaults.CONFIG_DIR


def _write(tmp_path: Path, text: str, name: str = "trial") -> Path:
    path = tmp_path / "pretrain" / f"{name}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_a_name_is_looked_up_and_a_path_is_kept() -> None:
    assert config_path("pretrain", "base-900m") == Path("configs/pretrain/base-900m.toml")
    assert config_path("pretrain", "my/run.toml") == Path("my/run.toml")


def test_keys_may_be_spelled_like_flags_and_out_defaults_to_the_name(tmp_path: Path) -> None:
    path = _write(tmp_path, "qk-norm = true\nbatch_size = 16\n")
    assert read_config(path) == {"out": "artifacts/runs/trial", "qk_norm": True, "batch_size": 16}
    assert read_config(_write(tmp_path, 'out = "elsewhere"\n'))["out"] == "elsewhere"


def test_a_missing_config_lists_the_available_ones(tmp_path: Path) -> None:
    _write(tmp_path, "", name="small-250m")
    assert available("pretrain", tmp_path) == ["small-250m"]
    with pytest.raises(FileNotFoundError, match="small-250m"):
        read_config(tmp_path / "pretrain" / "nope.toml")


def test_the_config_fills_defaults_and_flags_still_win(tmp_path: Path) -> None:
    path = _write(tmp_path, 'preset = "base"\nrope = true\nlr = 2e-3\ntokens = "850M"\n')
    args = parse_args(["pretrain", "--config", str(path), "--lr", "5e-4", "--no-rope"])
    assert (args.preset, args.tokens, args.out) == ("base", "850M", "artifacts/runs/trial")
    assert (args.lr, args.rope) == (5e-4, False)


def test_without_a_config_nothing_changes() -> None:
    args = parse_args(["pretrain"])
    assert (args.config, args.preset, args.rope, args.out) == (
        None,
        "small",
        False,
        defaults.PRETRAIN_RUN,
    )


@pytest.mark.parametrize("text", ["lerning_rate = 1e-3\n", 'preset = "huge"\n'])
def test_unknown_options_and_bad_choices_are_usage_errors(tmp_path: Path, text: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["pretrain", "--config", str(_write(tmp_path, text))])


def test_finetune_needs_init_or_scratch_from_somewhere() -> None:
    with pytest.raises(SystemExit, match="--init or --scratch"):
        args = parse_args(["finetune"])
        args.run(args)


@pytest.mark.parametrize("command", ["pretrain", "finetune"])
def test_the_shipped_configs_are_valid(command: str) -> None:
    names = available(command, REPO_CONFIGS)
    assert names
    for name in names:
        parser = build_parser().parse_args([command]).parser
        apply_config(parser, read_config(config_path(command, name, REPO_CONFIGS)))
