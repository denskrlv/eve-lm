import argparse
import glob
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from colibri import defaults
from colibri.config import PRESETS, STOP, ModelConfig
from colibri.progress import style
from colibri.runtime import pick_device
from colibri.train.loop import DTYPES, TrainConfig

if TYPE_CHECKING:
    from colibri.data.schema import Tool
    from colibri.infer.predict import Answers
    from colibri.tokenizer.bpe import BPETokenizer

TRAIN_DEFAULTS = TrainConfig(steps=1)  # the CLI's defaults are TrainConfig's own
DTYPE_CHOICES = list(DTYPES)
FROZEN_LR, FULL_LR = 1e-3, 3e-4  # finetune's default lr with and without --freeze
ALL_VAL_BATCHES = 10_000  # finetune evaluates on all of val: it is small
FINETUNE_WARMUP = 200

INTENTS_SHOWN, TOOLS_SHOWN = 4, 3  # answers colibri ask shows for the intent and per call
Sample = str | list[str]  # a text, or a tool schema as its parts

type Subcommands = argparse._SubParsersAction[argparse.ArgumentParser]


class _StdoutHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes plain lines to whatever ``sys.stdout`` is when a record arrives, like ``print``
    (pytest and redirects swap it). Warnings and errors start with their level: "warning: …"."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


def setup_logging() -> None:
    """Show the library's status lines (the ``colibri`` loggers, INFO and up) on stdout."""
    logger = logging.getLogger("colibri")
    logger.handlers = [_StdoutHandler()]  # replaced, not added: main() may run more than once
    logger.setLevel(logging.INFO)
    logger.propagate = False


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
    """Print the torch version, the device, and each preset's size."""
    print(f"torch {torch.__version__}, device: {pick_device()}")
    print(f"{'preset':<8}{'d_model':>9}{'heads':>7}{'enc':>5}{'dec':>5}{'q':>4}{'params':>10}")
    for name, cfg in PRESETS.items():
        print(
            f"{name:<8}{cfg.d_model:>9}{cfg.n_heads:>7}{cfg.n_encoder_layers:>5}"
            f"{cfg.n_decoder_layers:>5}{cfg.n_question_layers:>4}{cfg.approx_params() / 1e6:>9.1f}M"
        )


def model(args: argparse.Namespace) -> None:
    """Learn (M3): parameters per part and memory size, like the book's count in ch. 4.6."""
    from colibri.config import preset
    from colibri.model.colibri import Colibri

    names = [args.preset] if args.preset else list(PRESETS)
    counts = {name: Colibri(preset(name)).parameter_counts() for name in names}
    print(f"{'':<12}" + "".join(f"{name:>12}" for name in names))
    for part in counts[names[0]]:
        print(f"{part:<12}" + "".join(f"{counts[n][part]:>12,}" for n in names))
    # float32: 4 bytes per parameter. The LM head adds nothing: it reuses the token embeddings.
    print(f"{'MB (fp32)':<12}" + "".join(f"{counts[n]['total'] * 4 / 2**20:>12.1f}" for n in names))


def tokenizer_train(args: argparse.Namespace) -> None:
    """Train the BPE tokenizer on web text plus weighted schemas and commands."""
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


def _encode_sample(tokenizer: "BPETokenizer", sample: Sample) -> list[int]:
    """A sample's token ids; a schema is a list of parts (see ``Tool.render_parts``)."""
    parts = [sample] if isinstance(sample, str) else sample
    return [i for part in parts for i in tokenizer.encode(part, specials=True)]


def _print_tokens_per_word(tokenizer: "BPETokenizer", samples: dict[str, Sequence[Sample]]) -> None:
    """A table of words and tokens per word for each kind of sample, then the vocab size."""
    print(f"{'':<22}{'words':>9}{'tokens/word':>13}")
    for name, texts in samples.items():
        words = sum(len(" ".join(t if isinstance(t, list) else [t]).split()) for t in texts)
        n_tokens = sum(len(_encode_sample(tokenizer, t)) for t in texts)
        print(f"{name:<22}{words:>9,}{n_tokens / words:>13.2f}")
    print(f"vocabulary: {tokenizer.vocab_size:,}")


def tokenizer_stats(args: argparse.Namespace) -> None:
    """Learn (M1): tokens per word on unseen web text, on commands and on tool schemas."""
    from colibri.data.corpus import iter_documents, take_bytes
    from colibri.data.jsonl import read_lines
    from colibri.data.schema import load_library
    from colibri.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer)
    docs = list(
        take_bytes(
            iter_documents(_parquet_files(args.parquet)), _count(args.text), _count(args.skip)
        )
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
        print("\n" + " | ".join(tokenizer.pieces(_encode_sample(tokenizer, text))))


def corpus(args: argparse.Namespace) -> None:
    """Encode the pretraining corpus into train and val token files."""
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


def _train_config(args: argparse.Namespace, steps: int, lr: float, **extra: Any) -> TrainConfig:
    """The ``TrainConfig`` from the options ``pretrain`` and ``finetune`` share."""
    return TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        lr=lr,
        warmup_steps=min(args.warmup, steps),
        dtype=args.dtype,
        seed=args.seed,
        eval_every=args.eval_every,
        save_every=args.eval_every,
        log_every=args.log_every,
        **extra,
    )


def _stop_note(args: argparse.Namespace) -> str:
    """The end of a run's banner: where ``--stop-after`` ends it, if given."""
    return f" · stopping at step {args.stop_after:,}" if args.stop_after else ""


def pretrain(args: argparse.Namespace) -> None:
    """Run span-corruption pretraining with the command line's options."""
    from colibri.config import preset
    from colibri.train.pretrain import pretrain as run
    from colibri.train.pretrain import steps_for

    overrides = {"rope": args.rope, "qk_norm": args.qk_norm}
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    cfg = preset(args.preset, **overrides)
    steps = args.steps or steps_for(_count(args.tokens), args.batch_size, cfg.max_encoder_len)
    train_cfg = _train_config(args, steps, args.lr, compile=args.compile)
    device = pick_device(args.device)
    print(
        f"{style('pretrain', '1')} {args.preset} · {cfg.approx_params() / 1e6:.1f}M params · "
        f"{device} · {args.dtype} · lr {args.lr:g} · dropout {cfg.dropout:g} · "
        + "".join(f"{name} · " for name in ("rope", "qk_norm") if getattr(cfg, name))
        + ("compiled · " if args.compile else "")
        + f"{steps:,} steps × {args.batch_size} windows of {cfg.max_encoder_len} tokens"
        + _stop_note(args)
    )
    run(
        cfg,
        train_cfg,
        args.data,
        args.tokenizer,
        args.out,
        device,
        resume=args.resume,
        stop_after=args.stop_after,
    )


def _finetune_banner(
    args: argparse.Namespace, device: torch.device, lr: float, steps: int, n_train: int
) -> str:
    """The line ``finetune`` prints before it starts: where from, how, and for how long."""

    start = args.init or f"{args.scratch} from scratch"
    return (
        f"{style('finetune', '1')} questions · from {start} · {device} · lr {lr:g} · "
        + ("body frozen · " if args.freeze else "")
        + f"{steps:,} steps × {args.batch_size} ({steps * args.batch_size / n_train:.1f} epochs)"
        + _stop_note(args)
    )


def finetune(args: argparse.Namespace) -> None:
    """Fine-tune the question heads with the command line's options."""
    from colibri.config import preset
    from colibri.data.generate.build import Sources
    from colibri.data.jsonl import read_jsonl
    from colibri.train.finetune import finetune as run

    n_train = len(read_jsonl(Path(args.task) / "train.jsonl"))
    steps = args.steps or round(args.epochs * n_train / args.batch_size)
    lr = args.lr or (FROZEN_LR if args.freeze else FULL_LR)
    train_cfg = _train_config(args, steps, lr, eval_batches=ALL_VAL_BATCHES)
    device = pick_device(args.device)
    print(_finetune_banner(args, device, lr, steps, n_train))
    run(
        train_cfg,
        args.task,
        args.out,
        device,
        Sources.load().library,
        init=args.init,
        scratch=preset(args.scratch) if args.scratch else None,
        tokenizer_path=args.tokenizer,
        freeze=args.freeze,
        overrides={} if args.dropout is None else {"dropout": args.dropout},
        resume=args.resume,
        stop_after=args.stop_after,
    )


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
        records = [json.loads(line) for line in path.read_text().splitlines()]
        start: dict[str, Any] = next((r for r in records if r["kind"] == "start"), {})
        vals = {r["step"]: r for r in records if r["kind"] == "val"}
        runs[Path(run).name] = (start, vals)
    return runs


def _print_settings(runs: Runs, width: int) -> None:
    """One row per setting in ``SETTINGS``, one column per run."""
    for label, get in SETTINGS:
        print(f"{label:<14}" + "".join(f"{get(s)!s:>{width}}" for s, _ in runs.values()))


def _cell(value: float | None, metric: str) -> str:
    """A metric value as compare shows it: a percentage for accuracies, blank if missing."""
    if value is None:
        return ""
    return f"{value:.1%}" if metric.endswith("acc") else f"{value:.4f}"


def _print_metric_tables(runs: Runs, metrics: list[str], width: int) -> None:
    """For each metric, one row per eval step that any run logged, one column per run."""
    steps = sorted({step for _, vals in runs.values() for step in vals})
    for metric in metrics:
        print(f"\n{metric}")
        for step in steps:
            cells = [_cell(vals.get(step, {}).get(metric), metric) for _, vals in runs.values()]
            print(f"  step {step:<8,}" + "".join(f"{c:>{width}}" for c in cells))


def compare(args: argparse.Namespace) -> None:
    """Learn (M4): val metrics of several runs side by side, one row per eval step."""
    runs = _read_runs(args.runs)
    width = max(12, *(len(name) for name in runs)) + 2
    print(f"{'':<14}" + "".join(f"{name:>{width}}" for name in runs))
    _print_settings(runs, width)
    _print_metric_tables(runs, args.metric, width)


def _print_row(label: str, value: str) -> None:
    """``  label      value``, the label dimmed in a terminal."""
    print(f"  {style(f'{label:<11}', '2', stream=sys.stdout)}{value}")


def _print_fill_report(
    model_path: str, meta: dict[str, Any], device: torch.device, filled: list[str]
) -> None:
    """Which checkpoint filled the holes and how far it was trained, then each filled text."""
    trained = f"{meta['step']:,} steps" if "step" in meta else "?"
    if "val_lm" in meta:
        trained += f" · val loss {meta['val_lm']:.3f}"
    print()
    _print_row("checkpoint", model_path)
    _print_row("trained", trained)
    _print_row("device", str(device))
    print()
    for i, line in enumerate(filled, 1):
        print(f"  {style(f'{i:>2}', '2', stream=sys.stdout)}  {line}")
    print()


def fill(args: argparse.Namespace) -> None:
    """Learn (M4): ask a pretrained model to fill the <x0>, <x1>… holes in a sentence."""
    from colibri import progress
    from colibri.model.checkpoint import load_checkpoint
    from colibri.train.pretrain import SAMPLES, fill_in

    texts = args.text or list(SAMPLES)
    if bad := [t for t in texts if "<x0>" not in t]:
        raise SystemExit(f"put a hole like <x0> in the text: {bad[0]!r}")
    device = pick_device(args.device)
    with progress.bar(len(texts), "fill", "prompts", leave=False) as bar:
        bar.set_stats(f"loading {Path(args.model).name}")
        ckpt = load_checkpoint(args.model, device)
        filled = []
        for text in texts:
            bar.set_stats("decoding")
            filled.append(fill_in(ckpt.model, ckpt.tokenizer, text, args.max_new, progress.marks()))
            bar.update()

    _print_fill_report(str(args.model), ckpt.meta, device, filled)


def _tools(args: argparse.Namespace) -> list["Tool"]:
    """The tools for ``colibri ask``: names from the library and/or schemas from JSON files."""
    from colibri.data.generate.build import Sources
    from colibri.data.schema import GENERATION_KEYS, Tool

    tools = []
    if args.tools:
        library = Sources.load().library
        names = [name.strip() for name in args.tools.split(",") if name.strip()]
        if unknown := [name for name in names if name not in library]:
            raise SystemExit(f"unknown tools {unknown}; see data/tools/*.json or use --schema")
        tools += [library[name] for name in names]
    for path in args.schema or []:
        data = json.loads(Path(path).read_text())
        rows = data["tools"] if isinstance(data, dict) else data  # a data/tools file, or a list
        tools += [
            Tool.from_dict({k: v for k, v in t.items() if k not in GENERATION_KEYS}) for t in rows
        ]
    if not tools:
        raise SystemExit("give the tools with --tools a,b,c and/or --schema FILE")
    return tools


def _ranked(probs: dict[str, float], count: int) -> str:
    """The ``count`` most likely answers with their probabilities, the first in bold."""
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:count]
    return " · ".join(
        (style(f"{k} {p:.0%}", "1", stream=sys.stdout) if i == 0 else f"{k} {p:.0%}")
        for i, (k, p) in enumerate(top)
    )


def _print_answers(text: str, answers: "Answers", as_json: bool) -> None:
    """The heads' answers for one request: one JSON line, or the intent and each call slot up
    to the first one that says STOP."""
    if as_json:
        print(json.dumps({"utterance": text, **answers.to_json()}))
        return
    intent_label = f"{'intent':<8}"
    print(
        f"  {style(intent_label, '2', stream=sys.stdout)} {_ranked(answers.intent, INTENTS_SHOWN)}"
    )
    for i, slot in enumerate(answers.slots, 1):
        slot_label = f"{f'call {i}':<8}"
        print(f"  {style(slot_label, '2', stream=sys.stdout)} {_ranked(slot, TOOLS_SHOWN)}")
        if max(slot, key=slot.__getitem__) == STOP:
            break


def _ask_interactively(show: Callable[[str], None]) -> None:
    """Read requests from the terminal and show the answers, until an empty line or Ctrl-D."""
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        show(text)


def ask(args: argparse.Namespace) -> None:
    """Learn (M7): what the question heads answer for a typed request, with probabilities."""
    from colibri.infer.predict import answer_questions
    from colibri.model.checkpoint import load_checkpoint

    tools = _tools(args)
    ckpt = load_checkpoint(args.model, pick_device(args.device))

    def show(text: str) -> None:
        answers = answer_questions(ckpt.model, ckpt.tokenizer, text, tools)
        _print_answers(text, answers, args.json)

    if args.text:
        for text in args.text:
            if not args.json:
                print(text)
            show(text)
        return
    names = ", ".join(t.name for t in tools)
    print(f"{args.model} · tools: {names}\ntype a request (empty line or Ctrl-D to quit)")
    _ask_interactively(show)


def data_build(args: argparse.Namespace) -> None:
    """Build the task splits and print how many examples each has."""
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
    """Convert downloaded xLAM and MASSIVE data and print the report."""
    from colibri.data.generate.external import convert_external

    report = convert_external(args.raw, args.out, args.seed)
    print(json.dumps(report, indent=2))
    if not report:
        print(f"nothing found in {args.raw}/xlam or {args.raw}/massive")


def data_batch(args: argparse.Namespace) -> None:
    """Write new scenario batches for chat writers to fill in."""
    from colibri.data.generate.build import Sources, make_batch, next_batch_name

    batches = Path(args.dir) / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    for _ in range(args.count):
        name = next_batch_name(batches)
        seed = int(name.split("-")[1])
        sources = Sources.load(generation_dir=args.dir)
        batch = make_batch(sources, name, args.n, args.split, seed, args.ask)
        (batches / f"{name}.json").write_text(json.dumps(batch, indent=1, ensure_ascii=False))
        print(f"{batches / name}.json: {args.n} {args.split} scenarios")


def data_check(args: argparse.Namespace) -> None:
    """Validate a chat-written batch; exit with 1 if a line is rejected or missing."""
    from colibri.data.generate.build import Sources, check_written
    from colibri.data.jsonl import read_jsonl

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


def _add_schedule_args(
    parser: argparse.ArgumentParser, lr: float | None, lr_help: str | None, warmup: int
) -> None:
    """Batch size, learning rate, warmup and precision: the options both training runs take."""
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=lr, help=lr_help)
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=TRAIN_DEFAULTS.dtype)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Seed, eval/log cadence, device, resuming and stopping: shared by both training runs."""
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS.seed)
    parser.add_argument(
        "--eval-every", type=int, default=TRAIN_DEFAULTS.eval_every, help="also how often to save"
    )
    parser.add_argument("--log-every", type=int, default=TRAIN_DEFAULTS.log_every)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--stop-after", type=int, default=None, help="stop early at this step")


def _add_model_commands(commands: Subcommands) -> None:
    commands.add_parser("info", help="show the device and model presets").set_defaults(run=info)
    parser = commands.add_parser("model", help="parameters per part of a preset (M3)")
    parser.set_defaults(run=model)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="default: all")


def _add_tokenizer_commands(commands: Subcommands) -> None:
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


def _add_corpus_command(commands: Subcommands) -> None:
    parser = commands.add_parser("corpus", help="encode the pretraining corpus to token ids (M1)")
    parser.set_defaults(run=corpus)
    parser.add_argument("--parquet", default=defaults.PARQUET_GLOB)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
    parser.add_argument("--out", default=defaults.BIN_DIR)
    parser.add_argument("--train-tokens", default="250M")
    parser.add_argument("--val-tokens", default="2M")
    parser.add_argument("--workers", type=int, default=None)


def _add_pretrain_command(commands: Subcommands) -> None:
    parser = commands.add_parser("pretrain", help="span-corruption pretraining (M4)")
    parser.set_defaults(run=pretrain)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--data", default=defaults.BIN_DIR)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
    parser.add_argument("--out", default=defaults.PRETRAIN_RUN)
    parser.add_argument("--tokens", default="250M", help="corpus tokens to read (sets --steps)")
    parser.add_argument("--steps", type=int, default=None, help="default: from --tokens")
    _add_schedule_args(
        parser, lr=TRAIN_DEFAULTS.lr, lr_help=None, warmup=TRAIN_DEFAULTS.warmup_steps
    )
    parser.add_argument("--dropout", type=float, default=None, help="default: the preset's (0.1)")
    parser.add_argument("--rope", action="store_true", help="rotary positions, no position tables")
    parser.add_argument("--compile", action="store_true", help="torch.compile: ~2x faster on MPS")
    parser.add_argument("--qk-norm", action="store_true", help="RMS-normalize queries and keys")
    _add_run_args(parser)


def _add_finetune_command(commands: Subcommands) -> None:
    parser = commands.add_parser("finetune", help="fine-tune the question heads (M7)")
    parser.set_defaults(run=finetune)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", help="a pretraining checkpoint or an earlier finetune's model.pt")
    start.add_argument("--scratch", choices=sorted(PRESETS), help="random init (experiment E1)")
    parser.add_argument("--freeze", action="store_true", help="train only the question block (E2)")
    parser.add_argument("--task", default=defaults.TASK_DIR)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE, help="for --scratch")
    parser.add_argument("--out", default=defaults.FINETUNE_RUN)
    parser.add_argument("--epochs", type=float, default=2.0, help="sets --steps")
    parser.add_argument("--steps", type=int, default=None, help="default: from --epochs")
    _add_schedule_args(
        parser, lr=None, lr_help="default: 1e-3 frozen, else 3e-4", warmup=FINETUNE_WARMUP
    )
    parser.add_argument("--dropout", type=float, default=None, help="default: the checkpoint's")
    _add_run_args(parser)


def _add_inspect_commands(commands: Subcommands) -> None:
    parser = commands.add_parser("ask", help="ask the question heads about a request (M7)")
    parser.set_defaults(run=ask)
    parser.add_argument("text", nargs="*", help="requests; none: type them one by one")
    parser.add_argument("--tools", help="tool names from data/tools: set_lights,set_thermostat")
    parser.add_argument(
        "--schema", action="append", help="a JSON file of tool schemas (repeatable)"
    )
    parser.add_argument("--model", default=defaults.FINETUNED_MODEL)
    parser.add_argument("--json", action="store_true", help="one JSON line per request")
    parser.add_argument("--device", default=None)

    parser = commands.add_parser("compare", help="val metrics of training runs side by side (M4)")
    parser.set_defaults(run=compare)
    parser.add_argument("runs", nargs="+", help="run directories, e.g. artifacts/runs/ab-*")
    parser.add_argument("--metric", nargs="+", default=["val_content", "val_content_acc", "val_lm"])

    parser = commands.add_parser("fill", help="fill <x0> holes with a pretrained model (M4)")
    parser.set_defaults(run=fill)
    parser.add_argument("text", nargs="*", help='e.g. "Paris is the capital of <x0>."')
    parser.add_argument("--model", default=defaults.PRETRAINED_MODEL)
    parser.add_argument(
        "--max-new", type=int, default=defaults.MAX_NEW_TOKENS, help="most tokens to write"
    )
    parser.add_argument("--device", default=None)


def _add_data_commands(commands: Subcommands) -> None:
    data = commands.add_parser("data", help="build the task dataset (M6)")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    parser = data_commands.add_parser("build", help="templates + chat-written -> data/task/*.jsonl")
    parser.set_defaults(run=data_build)
    parser.add_argument("--out", default=defaults.TASK_DIR)
    parser.add_argument("--templates", default="80k", help="template examples for train")
    parser.add_argument("--heldout", default="2k", help="template examples with held-out tools")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
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


def build_parser() -> argparse.ArgumentParser:
    """The ``colibri`` command line: one subcommand per stage, each with a ``run`` function."""
    parser = argparse.ArgumentParser(prog="colibri")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_model_commands(commands)
    _add_tokenizer_commands(commands)
    _add_corpus_command(commands)
    _add_pretrain_command(commands)
    _add_finetune_command(commands)
    _add_inspect_commands(commands)
    _add_data_commands(commands)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the chosen subcommand."""
    args = build_parser().parse_args(argv)
    setup_logging()
    args.run(args)
