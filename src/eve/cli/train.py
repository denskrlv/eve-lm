"""Commands that train and compare runs: ``run`` (an experiment file) and ``compare``."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from eve.cli.common import Subcommands
from eve.progress import console, format_metric, higher_is_better, panel, table

if TYPE_CHECKING:
    from rich.table import Table


def _run_experiment(values: dict[str, Any], sets: list[str]) -> None:
    """Apply the ``--set`` overrides to an experiment's sections and run it."""
    from eve.experiment import Experiment, apply_sets, run

    try:
        experiment = Experiment.from_dict(apply_sets(values, sets))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"bad experiment: {error}") from error
    try:
        run(experiment)
    except FileExistsError as error:  # a new run over a saved one
        raise SystemExit(str(error)) from None


def run_file(args: argparse.Namespace) -> None:
    """Run an experiment file: any registered components, see ``eve.experiment``."""
    from eve.experiment import read

    path = Path(args.experiment)
    if not path.is_file():
        raise SystemExit(f"no experiment file {path}")
    items = [("config", str(path))] + ([("set", " · ".join(args.set))] if args.set else [])
    console.print(panel("eve run", items))
    _run_experiment(read(path), args.set)


Runs = dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]]  # name -> (start, evals)


SETTINGS: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [  # rows of compare's header
    ("lr", lambda s: s.get("train_config", {}).get("lr")),
    ("dropout", lambda s: s.get("model_config", {}).get("dropout")),
    ("rope", lambda s: s.get("model_config", {}).get("rope", False)),
    ("qk_norm", lambda s: s.get("model_config", {}).get("qk_norm", False)),
]


def _read_runs(run_dirs: list[str]) -> Runs:
    """Each run's start record and its val records by step, from its ``metrics.jsonl``."""
    runs: Runs = {}
    for run in run_dirs:
        path = Path(run) / "metrics.jsonl"
        if not path.exists():
            raise SystemExit(f"no metrics in {run}")
        try:
            lines = path.read_text().splitlines()
        except OSError as error:
            raise SystemExit(f"{path}: {error.strerror}") from None
        records = []
        for n, line in enumerate(lines, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f"{path}: line {n} isn't JSON") from None
        start: dict[str, Any] = next((r for r in records if r["kind"] == "start"), {})
        vals = {r["step"]: r for r in records if r["kind"] == "val"}
        runs[Path(run).name] = (start, vals)
    return runs


def _metric_cells(metric: str, values: list[float | None]) -> list[Text]:
    """One step's values of ``metric``, as in the training log; the best run's in bold green
    when more than one run has a value, a missing one blank."""
    present = [v for v in values if v is not None]
    pick = max if higher_is_better(metric) else min
    best = pick(present) if len(present) > 1 else None
    return [
        Text("") if v is None else Text(format_metric(metric, v), style="bold green" * (v == best))
        for v in values
    ]


def _compare_table(runs: Runs, metrics: list[str]) -> "Table":
    """Settings rows, then a section per metric with one row per eval step any run logged; one
    column per run."""
    out = table("", *runs, first_style="dim")
    for label, get in SETTINGS:
        out.add_row(label, *(str(get(start)) for start, _ in runs.values()))
    steps = sorted({step for _, vals in runs.values() for step in vals})
    for metric in metrics:
        out.add_section()
        out.add_row(Text(metric, style="bold cyan"))
        for step in steps:
            values = [vals.get(step, {}).get(metric) for _, vals in runs.values()]
            out.add_row(f"step {step:,}", *_metric_cells(metric, values))
    return out


def compare(args: argparse.Namespace) -> None:
    """Val metrics of several runs side by side, one row per eval step."""
    console.print(_compare_table(_read_runs(args.runs), args.metric))


def _add_set_arg(parser: argparse.ArgumentParser) -> None:
    """``--set section.key=value``: change any experiment value, e.g. train.lr=1e-3."""
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override an experiment value (repeatable), e.g. train.lr=1e-3",
    )


def add_run(commands: Subcommands) -> None:
    """Register ``eve run``."""
    parser = commands.add_parser("run", help="run an experiment file, e.g. a run's config.toml")
    parser.set_defaults(run=run_file)
    parser.add_argument("experiment", help="configs/experiments/*.toml, or <run>/config.toml")
    _add_set_arg(parser)


def add_compare(commands: Subcommands) -> None:
    """Register ``eve compare``."""
    parser = commands.add_parser("compare", help="val metrics of training runs side by side")
    parser.set_defaults(run=compare)
    parser.add_argument("runs", nargs="+", help="run directories, e.g. artifacts/runs/ab-*")
    parser.add_argument("--metric", nargs="+", default=["val_content", "val_content_acc", "val_lm"])
