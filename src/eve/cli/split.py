"""``eve split``: the splitter's labels, thresholds, evaluation and answers."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from eve import defaults
from eve.cli.common import Subcommands, add_device_arg, add_json_arg
from eve.progress import console, table
from eve.runtime import pick_device

if TYPE_CHECKING:
    from eve.model.checkpoint import Checkpoint
    from eve.model.splitter import Splitter
    from eve.split.decide import Thresholds

THRESHOLDS_FILE = "thresholds.json"  # in the splitter's run directory, from `eve split tune`
SHOWN_SCORES = (
    ("Exact", "exact_acc"),
    ("Tools", "tools_acc"),
    ("Values", "value_acc"),
    ("Full", "full_acc"),
)


def split_labels(args: argparse.Namespace) -> None:
    """Print a random sample of multi-action rows and each action's text, for checking the
    derived labels by hand."""
    from eve.data.generate.actions import row_actions
    from eve.data.jsonl import read_jsonl
    from eve.split.words import action_text, word_spans, words_in

    rows = [
        (row, actions)
        for row in read_jsonl(args.path)
        if (args.source is None or row.get("source") == args.source)
        and (actions := row_actions(row)) is not None
        and len(actions) > 1
    ]
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(rows))[: args.sample]
    for n, i in enumerate(picked, 1):
        row, actions = rows[int(i)]
        text = row["utterance"]
        words = word_spans(text)
        console.print(f"[bold]{n}[/bold]  {text}", markup=True, highlight=False)
        for k, action in enumerate(actions, 1):
            print(f"    {k}. {action_text(text, words, words_in(words, action))}")
    console.print(f"[dim]{len(picked)} of {len(rows):,} multi-action rows[/dim]")


def _splitter(run: str, device: str) -> "Checkpoint[Splitter]":
    from eve.model.checkpoint import model_file
    from eve.model.splitter import load_splitter

    try:
        return load_splitter(model_file(run), pick_device(device))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


def _run_dir(path: str) -> Path:
    """The run directory of a run directory or of a checkpoint file in one."""
    return Path(path) if Path(path).is_dir() else Path(path).parent


def _thresholds(run: str) -> "Thresholds":
    from eve.split.decide import Thresholds

    path = _run_dir(run) / THRESHOLDS_FILE
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run `eve split tune {run}` first")
    return Thresholds.from_json(json.loads(path.read_text()))


def split_tune(args: argparse.Namespace) -> None:
    """Pick the thresholds on the run's own val split and write them beside the model."""
    from eve.experiment import read
    from eve.split.decide import SplitCase, pick_thresholds, split_probs
    from eve.tasks.split import split_data

    ckpt = _splitter(args.splitter, args.device)
    config: dict[str, Any] = read(_run_dir(args.splitter) / "config.toml")
    data = split_data(
        ckpt.tokenizer,
        ckpt.model.cfg,
        int(config.get("train", {}).get("seed", 0)),
        **config["data"],
    )
    probs = split_probs(ckpt, [item.text for item in data.val])
    cases = [
        SplitCase(p[0], p[1], p[2], item.text, item.members)
        for item, p in zip(data.val, probs, strict=True)
        if p is not None
    ]
    th = pick_thresholds(cases, args.max_wrong_split)
    out = _run_dir(args.splitter) / THRESHOLDS_FILE
    out.write_text(json.dumps(th.to_json()) + "\n")
    print(f"exists {th.exists} · member {th.member} → {out}")


def _set_table(
    name: str, results: dict[str, dict[str, dict[str, float]]], splitter: dict[str, float]
) -> Any:
    out = table(name, "mode", "calls", *(label for label, _ in SHOWN_SCORES), first_style="bold")
    for mode, groups in results.items():
        for group, scores in groups.items():
            cells = [f"{scores[key]:.1%}" if key in scores else "–" for _, key in SHOWN_SCORES]
            out.add_row(mode, group, *cells)
    out.title = (
        f"splitter: exact {splitter['exact_split']:.1%} · unsure {splitter['unsure']:.1%} · "
        f"wrong split {splitter['wrong_split']:.1%}"
    )
    out.title_justify, out.title_style = "left", "dim"
    return out


def split_evaluate(args: argparse.Namespace) -> None:
    """Eve alone and split-then-call on whole eval sets, by number of gold calls; the
    splitter's own scores on each set's labeled rows and on the hard set."""
    from eve import progress
    from eve.data.generate.build import Sources
    from eve.data.jsonl import read_jsonl
    from eve.evaluate import load_questions_model, set_files
    from eve.split.pipeline import evaluate_pipeline, hard_set_scores, split_metrics

    th = _thresholds(args.splitter)
    splitter = _splitter(args.splitter, args.device)
    try:
        paths = set_files(args.evals, args.sets)
        eve = load_questions_model(args.model, pick_device(args.device))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    library = Sources.load().library
    out: dict[str, Any] = {"thresholds": th.to_json(), "sets": {}}
    total = sum(len(read_jsonl(p)) for p in paths.values())
    with progress.bar(total, "evaluate", "requests", leave=False) as bar:
        for name, path in paths.items():
            bar.set_stats(name)
            pipeline = evaluate_pipeline(eve, splitter, th, path, library, lambda: bar.update())
            out["sets"][name] = {
                "pipeline": pipeline,
                "splitter": split_metrics(splitter, th, read_jsonl(path)),
            }
    if args.hard:
        out["hard"] = hard_set_scores(splitter, th, read_jsonl(args.hard))
    if args.json:
        print(json.dumps(out))
        return
    for name, result in out["sets"].items():
        console.print(_set_table(name, result["pipeline"], result["splitter"]))
    if "hard" in out:
        console.print(" · ".join(f"{k} {v:.0%}" for k, v in out["hard"].items()))
    console.print(
        "[dim]Values are free-decoded and compare between the two modes only, not with "
        "`eve evaluate`.[/dim]",
        markup=True,
    )


def split_ask(args: argparse.Namespace) -> None:
    """The actions the splitter cuts a request into (the request itself if it doesn't)."""
    from eve.split.decide import split_request

    actions = split_request(
        _splitter(args.splitter, args.device), args.text, _thresholds(args.splitter)
    )
    if len(actions) == 1:
        print(f"not split: {actions[0]}")
    for k, action in enumerate(actions if len(actions) > 1 else (), 1):
        print(f"{k}. {action}")


def add_split(commands: Subcommands) -> None:
    """Register ``eve split labels``, ``tune``, ``evaluate`` and ``ask``."""
    split = commands.add_parser("split", help="the splitter: labels, thresholds, evaluation")
    split_commands = split.add_subparsers(dest="split_command", required=True)
    parser = split_commands.add_parser("labels", help="print derived split labels to check")
    parser.set_defaults(run=split_labels)
    parser.add_argument("--path", default=f"{defaults.TASK_DIR}/train.jsonl")
    parser.add_argument("--source", default=None, help="e.g. xlam or template")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    parser = split_commands.add_parser("tune", help="pick the split thresholds on the run's val")
    parser.set_defaults(run=split_tune)
    parser.add_argument("splitter", help="the splitter's run directory")
    parser.add_argument("--max-wrong-split", type=float, default=0.02)
    add_device_arg(parser)

    from eve.evaluate import SETS

    parser = split_commands.add_parser("evaluate", help="Eve alone vs split then call")
    parser.set_defaults(run=split_evaluate)
    parser.add_argument("splitter", help="the splitter's run directory (tuned)")
    parser.add_argument("--model", default=defaults.FINETUNED_MODEL, help="Eve: model.pt or run")
    parser.add_argument("--evals", default=defaults.EVALS_DIR)
    parser.add_argument("--sets", nargs="+", default=list(SETS))
    parser.add_argument(
        "--hard", default=None, help="a hard set, e.g. data/generation/split-hard.jsonl"
    )
    add_json_arg(parser, "the results as one JSON object")
    add_device_arg(parser)

    parser = split_commands.add_parser("ask", help="the actions of one request")
    parser.set_defaults(run=split_ask)
    parser.add_argument("splitter", help="the splitter's run directory (tuned)")
    parser.add_argument("text")
    add_device_arg(parser)
