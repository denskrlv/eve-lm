"""``eve split``: the splitter's labels, thresholds, evaluation and answers."""

import argparse

import numpy as np

from eve import defaults
from eve.cli.common import Subcommands
from eve.progress import console


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


def add_split(commands: Subcommands) -> None:
    """Register ``eve split labels``."""
    split = commands.add_parser("split", help="the splitter: labels, thresholds, evaluation")
    split_commands = split.add_subparsers(dest="split_command", required=True)
    parser = split_commands.add_parser("labels", help="print derived split labels to check")
    parser.set_defaults(run=split_labels)
    parser.add_argument("--path", default=f"{defaults.TASK_DIR}/train.jsonl")
    parser.add_argument("--source", default=None, help="e.g. xlam or template")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
