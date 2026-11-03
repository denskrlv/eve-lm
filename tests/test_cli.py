"""Every ``eve`` subcommand runs: its ``--help``, and a smoke run of the commands no other test
module runs. The checks are on exit status and a few stable facts, not on whole outputs."""

import argparse
import json
import shlex
from pathlib import Path

import pytest
import torch
from helpers import examples, save_joint, save_retriever

from eve.cli import build_parser, main
from eve.config import preset
from eve.experiment import Experiment, apply_sets, read
from eve.model.checkpoint import save_checkpoint
from eve.model.eve import Eve
from eve.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).parents[1]


def command_paths(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> list[list[str]]:
    """Every command path under ``parser``: ``["data"]``, ``["data", "check"]``…"""
    paths = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                paths.append([*prefix, name])
                paths += command_paths(sub, (*prefix, name))
    return paths


def test_every_subcommand_has_help(capsys: pytest.CaptureFixture[str]) -> None:
    paths = command_paths(build_parser())
    assert len(paths) >= 20
    for path in paths:
        with pytest.raises(SystemExit) as exit_:
            main([*path, "--help"])
        assert exit_.value.code == 0, path
        assert "usage: eve" in capsys.readouterr().out


def test_fill_writes_into_the_holes(
    tmp_path: Path, tokenizer: BPETokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "model.pt", Eve(preset("tiny")), tokenizer, {"step": 3})
    options = ["--model", str(tmp_path / "model.pt"), "--max-new", "2", "--device", "cpu"]

    main(["fill", "Paris is the capital of <x0>.", *options])

    out = capsys.readouterr().out
    assert "eve fill" in out and "3 steps" in out


def test_fill_needs_a_hole() -> None:
    with pytest.raises(SystemExit, match="put a hole like <x0>"):
        main(["fill", "no hole here"])


def test_export_command_prints_the_upload_line(
    tmp_path: Path, tokenizer: BPETokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    joint = save_joint(tmp_path / "joint.pt", tokenizer)

    main(["export", str(joint), "--out", str(tmp_path / "out"), "--repo", "me/eve"])

    assert (tmp_path / "out" / "model.pt").is_file()
    assert "hf upload me/eve" in capsys.readouterr().out


def test_retriever_eval_scores_bm25_and_a_retriever(
    tmp_path: Path, library: dict, tokenizer: BPETokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    lines = [json.dumps(e.to_json()) + "\n" for e in examples(library)]
    (evals / "val.jsonl").write_text("".join(lines))
    retriever = save_retriever(tmp_path / "retriever.pt", tokenizer)
    args = ["--evals", str(evals), "--splits", "val", "--sizes", "10", "--fill", "random"]

    main(["retriever", "eval", str(retriever), *args, "--json", "--device", "cpu"])
    results = json.loads(capsys.readouterr().out)
    assert set(results) == {"bm25", "dense"}
    assert "mrr" in results["bm25"]["val@10"]

    main(["retriever", "eval", *args])
    assert "eve retriever eval" in capsys.readouterr().out


def test_corpus_fetch_refuses_unknown_sources() -> None:
    with pytest.raises(SystemExit, match="unknown sources"):
        main(["corpus", "fetch", "nope"])


def test_ask_reports_a_bad_schema_file(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "no-tools.json").write_text(json.dumps({"x": 1}))
    for name in ("missing.json", "broken.json", "no-tools.json"):
        path = str(tmp_path / name)
        with pytest.raises(SystemExit) as exit_:
            main(["ask", "hi", "--schema", path])
        assert str(exit_.value.code).startswith(path), name


def test_data_check_reports_a_missing_batch(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="batch-9999"):
        main(["data", "check", "batch-9999", "--dir", str(tmp_path)])


def test_compare_reports_an_unreadable_metrics_file(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"kind": "val", "step": 1}\nnot json\n')
    with pytest.raises(SystemExit, match=r"metrics\.jsonl: line 2 "):
        main(["compare", str(run)])


def test_fill_and_retriever_eval_refuse_the_wrong_kind_of_model(
    tmp_path: Path, tokenizer: BPETokenizer
) -> None:
    retriever = str(save_retriever(tmp_path / "retriever.pt", tokenizer))
    joint = str(save_joint(tmp_path / "joint.pt", tokenizer))
    with pytest.raises(SystemExit, match="not Eve"):
        main(["fill", "a <x0>", "--model", retriever, "--device", "cpu"])
    with pytest.raises(SystemExit, match="isn't a retriever"):
        main(["retriever", "eval", joint, "--splits", "val", "--device", "cpu"])


def test_compare_reports_a_metrics_file_it_cannot_read(tmp_path: Path) -> None:
    (tmp_path / "run-a" / "metrics.jsonl").mkdir(parents=True)  # a directory, not a file
    with pytest.raises(SystemExit, match="metrics.jsonl"):
        main(["compare", str(tmp_path / "run-a")])


def readme_commands() -> list[list[str]]:
    """Each ``uv run eve …`` line of the README's code blocks, as ``eve``'s arguments."""
    text = (ROOT / "README.md").read_text().replace("\\\n", " ")  # join continued lines
    commands = [line.split("#")[0] for line in text.splitlines() if line.startswith("uv run eve ")]
    return [shlex.split(command)[3:] for command in commands]


def test_the_readme_commands_exist() -> None:
    commands = readme_commands()
    assert len(commands) >= 10
    for argv in commands:
        args = build_parser().parse_args(argv)
        if args.command == "run":  # the experiment file exists and takes the overrides
            Experiment.from_dict(apply_sets(read(ROOT / args.experiment), args.set))
