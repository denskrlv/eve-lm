"""Commands that build data: ``tokenizer``, ``corpus`` and ``data``."""

import argparse
import glob
import json
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
    """Learn (M1): tokens per word on unseen web text, on commands and on tool schemas."""
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

    if args.mix and not 0 < args.mix_share < 1:
        raise SystemExit(f"--mix-share must be between 0 and 1, got {args.mix_share}")
    meta = encode_corpus(
        _parquet_files(args.parquet),
        args.tokenizer,
        args.out,
        train_tokens=count(args.train_tokens),
        val_tokens=count(args.val_tokens),
        workers=args.workers,
        mix=_parquet_files(args.mix) if args.mix else (),
        mix_share=args.mix_share,
        seed=args.seed,
    )
    mixed = ""
    if mix := meta.get("mix"):
        mixed = f" ({mix['byte_share']:.0%} mixed in from {mix['documents']:,} documents)"
    success(
        f"train {meta['train_tokens']:,}{mixed} and val {meta['val_tokens']:,} tokens", args.out
    )


def _tinytools_names(names: list[str]) -> list[str]:
    """The named TinyTools sources, or all of them; exits on an unknown name."""
    from eve.data.tinytools.build import SOURCES

    if unknown := [n for n in names if n not in SOURCES]:
        raise SystemExit(f"unknown sources {unknown}; choose from {sorted(SOURCES)}")
    return names or list(SOURCES)


def corpus_fetch(args: argparse.Namespace) -> None:
    """Download the TinyTools sources into the raw folder."""
    from eve.data.tinytools.build import SOURCES
    from eve.data.tinytools.fetch import fetch

    for name in _tinytools_names(args.sources):
        saved = fetch(SOURCES[name].downloads, Path(args.raw), args.force)
        success(f"{name}: {len(saved)} downloaded" if saved else f"{name}: already here")


def corpus_tinytools(args: argparse.Namespace) -> None:
    """Convert the downloaded TinyTools sources into one parquet file each."""
    from eve.data.tinytools.build import build_tinytools

    names = _tinytools_names(args.sources)
    report = build_tinytools(names, args.raw, args.out, args.tokenizer, args.seed)
    built = {n: report["sources"][n] for n in names}
    if lacking := [n for n, r in built.items() if "missing" in r]:
        warn(f"not downloaded yet: {', '.join(lacking)} (run `eve corpus fetch`)")
    console.print(tree("TinyTools", {n: r for n, r in built.items() if "missing" not in r}))
    if not report["eval_sets"]:
        warn("no eval sets found to dedupe against: run `eve data build` first")
    success(f"{report['tokens']:,} tokens in all", Path(args.out) / "report.json")


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
    if rejected := report["rejected_chat_lines"]:
        warn(f"{rejected} chat-written lines rejected (see `eve data check`)")
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


def data_batch(args: argparse.Namespace) -> None:
    """Write new scenario batches for chat writers to fill in."""
    from eve.data.generate.build import Sources, make_batch, next_batch_name

    batches = Path(args.dir) / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    for _ in range(args.count):
        name = next_batch_name(batches)
        seed = int(name.split("-")[1])
        sources = Sources.load(generation_dir=args.dir)
        batch = make_batch(sources, name, args.n, args.split, seed, args.ask)
        (batches / f"{name}.json").write_text(json.dumps(batch, indent=1, ensure_ascii=False))
        success(f"{args.n} {args.split} scenarios", batches / f"{name}.json")


def data_check(args: argparse.Namespace) -> None:
    """Validate a chat-written batch; exit with 1 if a line is rejected or missing."""
    from eve.data.generate.build import Sources, check_written
    from eve.data.jsonl import read_jsonl

    batch = json.loads((Path(args.dir) / "batches" / f"{args.batch}.json").read_text())
    written_path = Path(args.dir) / "written" / f"{args.batch}.jsonl"
    written = read_jsonl(written_path) if written_path.exists() else []
    examples, rejects = check_written(batch, written, Sources.load(generation_dir=args.dir).library)
    missing = {s["id"] for s in batch["scenarios"]} - {line.get("id") for line in written}
    if rejects:
        console.print(_rejects_table(rejects))
    counts = f"{len(examples)} ok · {len(rejects)} rejected · {len(missing)} scenarios not written"
    if rejects or missing:
        console.print(Text(f"✗ {counts}", style="bold red"))
        raise SystemExit(1)
    success(counts)


def _rejects_table(rejects: list[tuple[dict[str, Any], list[str]]]) -> "Table":
    """One row per rejected line: its id, its utterance and why it was rejected."""
    out = table("id", "utterance", "errors", first_style="dim")
    for column in out.columns[1:]:  # text, not numbers: left-aligned and wrapped
        column.justify, column.no_wrap = "left", False
    for line, errors in rejects:
        out.add_row(str(line.get("id")), str(line.get("utterance")), Text("\n".join(errors), "red"))
    return out


def add_tokenizer(commands: Subcommands) -> None:
    tok = commands.add_parser("tokenizer", help="train or inspect the BPE tokenizer (M1)")
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
    group = commands.add_parser("corpus", help="the pretraining corpus")
    corpus_commands = group.add_subparsers(dest="corpus_command", required=True)
    parser = corpus_commands.add_parser("encode", help="encode parquet text to token ids (M1)")
    parser.set_defaults(run=corpus_encode)
    parser.add_argument("--parquet", default=defaults.PARQUET_GLOB)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
    parser.add_argument("--out", default=defaults.BIN_DIR)
    parser.add_argument("--train-tokens", default="900M")
    parser.add_argument("--val-tokens", default="2M")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--mix", help="a glob of parquet files (a text column) to spread through the train tokens"
    )
    parser.add_argument(
        "--mix-share", type=float, default=0.3, help="their share of the train text (with --mix)"
    )
    parser.add_argument("--seed", type=int, default=0, help="the order of the mixed documents")
    parser = corpus_commands.add_parser("fetch", help="download the TinyTools sources")
    parser.set_defaults(run=corpus_fetch)
    parser.add_argument("sources", nargs="*", help="default: all")
    parser.add_argument("--raw", default=defaults.RAW_DIR)
    parser.add_argument("--force", action="store_true", help="download again what is there")
    parser = corpus_commands.add_parser(
        "tinytools", help="convert the TinyTools sources -> data/tinytools/*.parquet"
    )
    parser.set_defaults(run=corpus_tinytools)
    parser.add_argument("sources", nargs="*", help="default: all")
    parser.add_argument("--raw", default=defaults.RAW_DIR)
    parser.add_argument("--out", default=defaults.TINYTOOLS_DIR)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE, help="counts the tokens")
    parser.add_argument("--seed", type=int, default=0)


def _add_data_build_command(data_commands: Subcommands) -> None:
    parser = data_commands.add_parser(
        "build", help="templates + chat-written -> data/task/train.jsonl, data/evals/*.jsonl"
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
    data = commands.add_parser("data", help="build the task dataset (M6)")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    _add_data_build_command(data_commands)
    parser = data_commands.add_parser("external", help="convert xLAM and MASSIVE -> data/external")
    parser.set_defaults(run=data_external)
    parser.add_argument("--raw", default=defaults.RAW_DIR)
    parser.add_argument("--out", default=defaults.EXTERNAL_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser = data_commands.add_parser("batch", help="new scenario batches for chat writers")
    parser.set_defaults(run=data_batch)
    parser.add_argument("--split", choices=["train", "val", "probe"], default="train")
    parser.add_argument("--n", type=int, default=50, help="scenarios per batch")
    parser.add_argument("--count", type=int, default=1, help="how many batches")
    parser.add_argument("--ask", type=float, default=None, help="share of ASK scenarios, e.g. 0.3")
    parser.add_argument("--dir", default=defaults.GENERATION_DIR)
    parser = data_commands.add_parser("check", help="validate a chat-written batch")
    parser.set_defaults(run=data_check)
    parser.add_argument("batch", help="e.g. batch-0001")
    parser.add_argument("--dir", default=defaults.GENERATION_DIR)
