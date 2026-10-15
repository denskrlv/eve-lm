import argparse
import glob
import json
import time
from pathlib import Path

import torch

from colibri.config import PRESETS
from colibri.runtime import pick_device

DEFAULT_PARQUET = "data/raw/fineweb-edu/*.parquet"
DEFAULT_TOKENIZER = "artifacts/tokenizer/colibri-8k.json"


def _count(text: str) -> int:
    """'250M' -> 250_000_000, '2k' -> 2000, '1000' -> 1000."""
    scale = {"k": 10**3, "m": 10**6, "b": 10**9}.get(text[-1].lower(), 1)
    return int(float(text[:-1] if scale > 1 else text) * scale)


def _parquet_files(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no parquet files match {pattern}")
    return paths


def info(_: argparse.Namespace) -> None:
    print(f"torch {torch.__version__}, device: {pick_device()}")
    print(f"{'preset':<8}{'d_model':>9}{'heads':>7}{'enc':>5}{'dec':>5}{'q':>4}{'params':>10}")
    for name, cfg in PRESETS.items():
        print(
            f"{name:<8}{cfg.d_model:>9}{cfg.n_heads:>7}{cfg.n_encoder_layers:>5}"
            f"{cfg.n_decoder_layers:>5}{cfg.n_question_layers:>4}{cfg.approx_params() / 1e6:>9.1f}M"
        )


def tokenizer_train(args: argparse.Namespace) -> None:
    from colibri.data.corpus import iter_documents, take_bytes, tokenizer_texts
    from colibri.tokenizer.bpe import BPETokenizer

    corpus = take_bytes(iter_documents(_parquet_files(args.parquet)), _count(args.text))
    start = time.time()
    tokenizer = BPETokenizer.train(
        tokenizer_texts(corpus, args.tools, args.commands, args.weight),
        vocab_size=args.vocab_size,
        progress=True,
    )
    tokenizer.save(args.out)
    print(f"{tokenizer.vocab_size} tokens in {time.time() - start:.0f}s -> {args.out}")


def tokenizer_stats(args: argparse.Namespace) -> None:
    """Learn (M1): tokens per word on unseen web text, on commands and on tool schemas."""
    from colibri.data.corpus import iter_documents, take_bytes
    from colibri.data.schema import load_library
    from colibri.tokenizer.bpe import BPETokenizer

    ours = BPETokenizer.load(args.tokenizer)
    docs = list(
        take_bytes(
            iter_documents(_parquet_files(args.parquet)), _count(args.text), _count(args.skip)
        )
    )
    tools = load_library(args.tools)
    commands = [
        line.strip()
        for line in Path(args.commands).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    schemas = [t.render_parts() for t in tools.values()]
    samples = {
        "web text (held out)": docs,
        "commands": commands,
        "tool schemas": schemas,
    }

    # A schema is a list of parts (see Tool.render_parts); other samples are plain strings.
    def ours_encode(t: str | list[str]) -> list[int]:
        parts = [t] if isinstance(t, str) else t
        return [i for part in parts for i in ours.encode(part, specials=True)]

    print(f"{'':<22}{'words':>9}{'tokens/word':>13}")
    for name, texts in samples.items():
        words = sum(len(" ".join(t if isinstance(t, list) else [t]).split()) for t in texts)
        n_ours = sum(len(ours_encode(t)) for t in texts)
        print(f"{name:<22}{words:>9,}{n_ours / words:>13.2f}")
    print(f"vocabulary: {ours.vocab_size:,}")

    for text in [commands[1], schemas[0]]:
        print("\n" + " | ".join(ours.pieces(ours_encode(text))))


def corpus(args: argparse.Namespace) -> None:
    from colibri.data.corpus import encode_corpus

    meta = encode_corpus(
        _parquet_files(args.parquet),
        args.tokenizer,
        args.out,
        train_tokens=_count(args.train_tokens),
        val_tokens=_count(args.val_tokens),
        workers=args.workers,
    )
    print(f"train {meta['train_tokens']:,} and val {meta['val_tokens']:,} tokens -> {args.out}")


def data_build(args: argparse.Namespace) -> None:
    from colibri.config import preset
    from colibri.data.generate.build import build
    from colibri.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer) if Path(args.tokenizer).exists() else None
    if tokenizer is None:
        print(f"no tokenizer at {args.tokenizer}: not checking example lengths")
    report = build(
        args.out,
        _count(args.templates),
        _count(args.heldout),
        args.seed,
        tokenizer,
        preset("small"),
    )
    for name, stats in report["splits"].items():
        print(f"{name:<18} {stats['examples']:>7,}  {stats['intent']}")
    print(f"rejected chat lines: {report['rejected_chat_lines']} (see `colibri data check`)")
    print(f"report -> {args.out}/report.json")


def data_external(args: argparse.Namespace) -> None:
    from colibri.data.generate.external import convert_external

    report = convert_external(args.raw, args.out, args.seed)
    print(json.dumps(report, indent=2))
    if not report:
        print(f"nothing found in {args.raw}/xlam or {args.raw}/massive")


def data_batch(args: argparse.Namespace) -> None:
    from colibri.data.generate.build import Sources, make_batch, next_batch_name

    batches = Path(args.dir) / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    for _ in range(args.count):
        name = next_batch_name(batches)
        seed = int(name.split("-")[1])
        batch = make_batch(Sources.load(generation_dir=args.dir), name, args.n, args.split, seed)
        (batches / f"{name}.json").write_text(json.dumps(batch, indent=1, ensure_ascii=False))
        print(f"{batches / name}.json: {args.n} {args.split} scenarios")


def data_check(args: argparse.Namespace) -> None:
    from colibri.data.generate.build import Sources, check_written, read_jsonl

    batch = json.loads((Path(args.dir) / "batches" / f"{args.batch}.json").read_text())
    written_path = Path(args.dir) / "written" / f"{args.batch}.jsonl"
    written = read_jsonl(written_path) if written_path.exists() else []
    examples, rejects = check_written(batch, written, Sources.load(generation_dir=args.dir).library)
    missing = {s["id"] for s in batch["scenarios"]} - {line.get("id") for line in written}
    for line, errors in rejects:
        print(f"{line.get('id')}: {line.get('utterance')!r}\n    " + "\n    ".join(errors))
    print(f"{len(examples)} ok, {len(rejects)} rejected, {len(missing)} scenarios not written")
    if rejects or missing:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="colibri")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="show the device and model presets").set_defaults(run=info)

    tok = commands.add_parser("tokenizer", help="train or inspect the BPE tokenizer (M1)")
    tok_commands = tok.add_subparsers(dest="tokenizer_command", required=True)
    for name, run, help_ in [
        ("train", tokenizer_train, "train the BPE on web text, tool schemas and commands"),
        ("stats", tokenizer_stats, "tokens per word and sample splits"),
    ]:
        p = tok_commands.add_parser(name, help=help_)
        p.set_defaults(run=run)
        p.add_argument("--parquet", default=DEFAULT_PARQUET)
        p.add_argument("--tools", default="data/tools")
        p.add_argument("--commands", default="data/tokenizer/commands.txt")
        if name == "train":
            p.add_argument("--text", default="50M", help="bytes of web text to train on")
            p.add_argument("--vocab-size", type=int, default=8192)
            p.add_argument("--weight", type=int, default=50, help="weight of schemas and commands")
            p.add_argument("--out", default=DEFAULT_TOKENIZER)
        else:
            p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
            p.add_argument("--text", default="5M", help="bytes of web text to measure on")
            p.add_argument("--skip", default="100M", help="bytes to skip (the training text)")

    p = commands.add_parser("corpus", help="encode the pretraining corpus to token ids (M1)")
    p.set_defaults(run=corpus)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--out", default="data/bin")
    p.add_argument("--train-tokens", default="250M")
    p.add_argument("--val-tokens", default="2M")
    p.add_argument("--workers", type=int, default=None)

    data = commands.add_parser("data", help="build the task dataset (M6)")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    p = data_commands.add_parser("build", help="templates + chat-written -> data/task/*.jsonl")
    p.set_defaults(run=data_build)
    p.add_argument("--out", default="data/task")
    p.add_argument("--templates", default="50k", help="template examples for train")
    p.add_argument("--heldout", default="2k", help="template examples with held-out tools")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p = data_commands.add_parser("external", help="convert xLAM and MASSIVE -> data/external")
    p.set_defaults(run=data_external)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/external")
    p.add_argument("--seed", type=int, default=0)
    p = data_commands.add_parser("batch", help="new scenario batches for chat writers")
    p.set_defaults(run=data_batch)
    p.add_argument("--split", choices=["train", "val", "probe"], default="train")
    p.add_argument("--n", type=int, default=50, help="scenarios per batch")
    p.add_argument("--count", type=int, default=1, help="how many batches")
    p.add_argument("--dir", default="data/generation")
    p = data_commands.add_parser("check", help="validate a chat-written batch")
    p.set_defaults(run=data_check)
    p.add_argument("batch", help="e.g. batch-0001")
    p.add_argument("--dir", default="data/generation")

    args = parser.parse_args(argv)
    args.run(args)
