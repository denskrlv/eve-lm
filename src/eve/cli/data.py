"""Commands that build data: ``tokenizer``, ``corpus`` and ``data``."""

import argparse
import glob
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from eve import defaults
from eve.cli.common import Subcommands, count
from eve.config import INTENTS, ModelConfig
from eve.progress import console, format_duration, rows, success, table, tree, warn

if TYPE_CHECKING:
    from rich.table import Table

    from eve.tokenizer.bpe import BPETokenizer


Sample = str | list[str]  # a text, or a tool schema as its parts


PIECE_COLORS = ("cyan", "magenta")  # tokenizer stats alternates them between token pieces


def _parquet_files(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no parquet files match {pattern}")
    return paths


def tokenizer_train(args: argparse.Namespace) -> None:
    """Train the BPE tokenizer on web text plus weighted schemas and commands."""
    from eve.data.corpus import iter_documents, take_bytes, tokenizer_texts
    from eve.tokenizer.bpe import BPETokenizer

    corpus = take_bytes(iter_documents(_parquet_files(args.parquet)), count(args.text))
    start = time.time()
    tokenizer = BPETokenizer.train(
        tokenizer_texts(corpus, args.tools, args.commands, args.weight),
        vocab_size=args.vocab_size,
        progress=True,
    )
    tokenizer.save(args.out)
    took = format_duration(time.time() - start)
    success(f"{tokenizer.vocab_size:,} tokens learned in {took}", args.out)


def _encode_sample(tokenizer: "BPETokenizer", sample: Sample) -> list[int]:
    """A sample's token ids; a schema is a list of parts (see ``Tool.render_parts``)."""
    parts = [sample] if isinstance(sample, str) else sample
    return [i for part in parts for i in tokenizer.encode(part, specials=True)]


def _print_tokens_per_word(tokenizer: "BPETokenizer", samples: dict[str, Sequence[Sample]]) -> None:
    """A table of words and tokens per word for each kind of sample, then the vocab size."""
    out = table("", "words", "tokens/word")
    for name, texts in samples.items():
        words = sum(len(" ".join(t if isinstance(t, list) else [t]).split()) for t in texts)
        n_tokens = sum(len(_encode_sample(tokenizer, t)) for t in texts)
        out.add_row(name, f"{words:,}", f"{n_tokens / words:.2f}")
    console.print(out)
    console.print(rows([("vocabulary", f"{tokenizer.vocab_size:,}")]))


def _pieces(pieces: list[str]) -> Text:
    """Token pieces in alternating colors, split by dim bars, so each token stands out."""
    return Text(" | ", style="dim").join(
        Text(piece, style=PIECE_COLORS[i % len(PIECE_COLORS)]) for i, piece in enumerate(pieces)
    )


def tokenizer_stats(args: argparse.Namespace) -> None:
    """Tokens per word on unseen web text, on commands and on tool schemas."""
    from eve.data.corpus import iter_documents, take_bytes
    from eve.data.jsonl import read_lines
    from eve.data.schema import load_library
    from eve.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer)
    docs = list(
        take_bytes(iter_documents(_parquet_files(args.parquet)), count(args.text), count(args.skip))
    )
    commands = read_lines(args.commands)
    schemas = [t.render_parts() for t in load_library(args.tools).values()]
    samples: dict[str, Sequence[Sample]] = {
        "web text (held out)": docs,
        "commands": commands,
        "tool schemas": schemas,
    }
    _print_tokens_per_word(tokenizer, samples)
    examples: list[Sample] = [commands[1], schemas[0]]
    for text in examples:
        console.print()
        console.print(_pieces(tokenizer.pieces(_encode_sample(tokenizer, text))))


def corpus_encode(args: argparse.Namespace) -> None:
    """Encode the pretraining corpus into train and val token files."""
    from eve.data.corpus import encode_corpus

    meta = encode_corpus(
        _parquet_files(args.parquet),
        args.tokenizer,
        args.out,
        train_tokens=count(args.train_tokens),
        val_tokens=count(args.val_tokens),
        workers=args.workers,
    )
    success(f"train {meta['train_tokens']:,} and val {meta['val_tokens']:,} tokens", args.out)


def data_build(args: argparse.Namespace) -> None:
    """Build the task splits and print how many examples each has."""
    from eve.config import preset
    from eve.data.generate.build import build
    from eve.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer) if Path(args.tokenizer).exists() else None
    if tokenizer is None:
        warn(f"no tokenizer at {args.tokenizer}: not checking example lengths")
    report = build(
        args.out,
        args.evals,
        args.templates,
        args.heldout,
        args.seed,
        tokenizer,
        preset("eve-32m"),
    )
    console.print(_splits_table(report["splits"]))
    success("built", Path(args.out) / "report.json")


def _splits_table(splits: dict[str, dict[str, Any]]) -> "Table":
    """Examples per split, in total and per intent."""

    out = table("split", "examples", *INTENTS)
    out.columns[0].style = "bold cyan"
    for name, stats in splits.items():
        intents = [f"{stats['intent'].get(intent, 0):,}" for intent in INTENTS]
        out.add_row(name, f"{stats['examples']:,}", *intents)
    return out


def data_external(args: argparse.Namespace) -> None:
    """Convert downloaded xLAM and MASSIVE data and print the report."""
    from eve.data.generate.external import convert_external

    report = convert_external(args.raw, args.out, args.seed)
    if not report:
        warn(f"nothing found in {args.raw}/xlam or {args.raw}/massive")
        return
    console.print(tree("external data", report))
    success("converted", Path(args.out) / "report.json")


def add_tokenizer(commands: Subcommands) -> None:
    """Register ``eve tokenizer train`` and ``eve tokenizer stats``."""
    tok = commands.add_parser("tokenizer", help="train or inspect the BPE tokenizer")
    tok_commands = tok.add_subparsers(dest="tokenizer_command", required=True)
    for name, run, help_ in [
        ("train", tokenizer_train, "train the BPE on web text, tool schemas and commands"),
        ("stats", tokenizer_stats, "tokens per word and sample splits"),
    ]:
        parser = tok_commands.add_parser(name, help=help_)
        parser.set_defaults(run=run)
        parser.add_argument("--parquet", default=defaults.PARQUET_GLOB)
        parser.add_argument("--tools", default=defaults.TOOLS_DIR)
        parser.add_argument("--commands", default=defaults.COMMANDS_FILE)
        if name == "train":
            parser.add_argument("--text", default="50M", help="bytes of web text to train on")
            parser.add_argument("--vocab-size", type=int, default=ModelConfig.vocab_size)
            parser.add_argument(
                "--weight",
                type=int,
                default=defaults.TOKENIZER_WEIGHT,
                help="weight of schemas and commands",
            )
            parser.add_argument("--out", default=defaults.TOKENIZER_FILE)
        else:
            parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
            parser.add_argument("--text", default="5M", help="bytes of web text to measure on")
            parser.add_argument("--skip", default="100M", help="bytes to skip (the training text)")


def add_corpus(commands: Subcommands) -> None:
    """Register ``eve corpus encode``."""
    group = commands.add_parser("corpus", help="the pretraining corpus")
    corpus_commands = group.add_subparsers(dest="corpus_command", required=True)
    parser = corpus_commands.add_parser("encode", help="encode parquet text to token ids")
    parser.set_defaults(run=corpus_encode)
    parser.add_argument("--parquet", default=defaults.PARQUET_GLOB)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
    parser.add_argument("--out", default=defaults.BIN_DIR)
    parser.add_argument("--train-tokens", default="900M")
    parser.add_argument("--val-tokens", default="2M")
    parser.add_argument("--workers", type=int, default=None)


def _add_data_build_command(data_commands: Subcommands) -> None:
    parser = data_commands.add_parser(
        "build", help="templates + external -> data/task/train.jsonl, data/evals/*.jsonl"
    )
    parser.set_defaults(run=data_build)
    parser.add_argument("--out", default=defaults.TASK_DIR, help="where train.jsonl goes")
    parser.add_argument("--evals", default=defaults.EVALS_DIR, help="where the eval sets go")
    parser.add_argument(
        "--templates",
        type=count,
        default=defaults.TEMPLATE_EXAMPLES,
        help="template examples for train",
    )
    parser.add_argument(
        "--heldout",
        type=count,
        default=defaults.HELDOUT_EXAMPLES,
        help="template examples with held-out tools",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)


def add_data(commands: Subcommands) -> None:
    """Register ``eve data build`` and ``external``."""
    data = commands.add_parser("data", help="build the task dataset")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    _add_data_build_command(data_commands)
    parser = data_commands.add_parser("external", help="convert xLAM and MASSIVE -> data/external")
    parser.set_defaults(run=data_external)
    parser.add_argument("--raw", default=defaults.RAW_DIR)
    parser.add_argument("--out", default=defaults.EXTERNAL_DIR)
    parser.add_argument("--seed", type=int, default=0)
