import argparse
import glob
import json
import logging
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rich.text import Text

from eve import defaults
from eve.config import INTENTS, PRESETS, STOP, ModelConfig
from eve.progress import (
    console,
    format_duration,
    format_metric,
    higher_is_better,
    panel,
    rows,
    success,
    table,
    tree,
    warn,
)
from eve.run_configs import apply_config, available, config_path, read_config
from eve.runtime import pick_device
from eve.train.config import DTYPES, TrainConfig

if TYPE_CHECKING:
    from rich.panel import Panel
    from rich.table import Table

    from eve.data.schema import Tool
    from eve.evaluate import SetResult
    from eve.infer.predict import Answers, WrittenCall
    from eve.tokenizer.bpe import BPETokenizer

TRAIN_DEFAULTS = TrainConfig(steps=1)  # the CLI's defaults are TrainConfig's own
DTYPE_CHOICES = list(DTYPES)
FROZEN_LR, FULL_LR = 1e-3, 3e-4  # finetune's default lr with and without --freeze
ALL_VAL_BATCHES = 10_000  # finetune evaluates on all of val: it is small
FINETUNE_WARMUP = 200

INTENTS_SHOWN, TOOLS_SHOWN = 4, 3  # answers eve ask shows for the intent and per call
SURE, UNSURE = 0.8, 0.5  # eve ask colors a top answer green from SURE, yellow from UNSURE
SCORE_COLUMNS = (  # eve evaluate's table: (label, key in the scores)
    ("macro", "macro_recall"),
    *((intent[:5], f"recall_{intent}") for intent in INTENTS),  # "unsupported" -> "unsup"
    ("tools", "tools_acc"),
    ("exact", "exact_acc"),
    ("values", "value_acc"),
    ("full", "full_acc"),
)
Sample = str | list[str]  # a text, or a tool schema as its parts
PIECE_COLORS = ("cyan", "magenta")  # tokenizer stats alternates them between token pieces

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
    """Show the library's status lines (the ``eve`` loggers, INFO and up) on stdout."""
    logger = logging.getLogger("eve")
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
    console.print(rows([("torch", torch.__version__), ("device", str(pick_device()))]))
    console.print()
    presets = table("preset", "d_model", "heads", "encoder", "decoder", "questions", "params")
    presets.columns[0].style = "bold cyan"
    for name, cfg in PRESETS.items():
        sizes = (cfg.d_model, cfg.n_heads, cfg.n_encoder_layers, cfg.n_decoder_layers)
        presets.add_row(
            name,
            *(str(size) for size in sizes),
            str(cfg.n_question_layers),
            f"{cfg.approx_params() / 1e6:.1f}M",
        )
    console.print(presets)


def model(args: argparse.Namespace) -> None:
    """Learn (M3): parameters per part and memory size, like the book's count in ch. 4.6."""
    from eve.config import preset
    from eve.model.eve import Eve

    names = [args.preset] if args.preset else list(PRESETS)
    counts = {name: Eve(preset(name)).parameter_counts() for name in names}
    parts = table("", *names, first_style="dim")
    for part in counts[names[0]]:
        if part == "total":
            parts.add_section()
        cells = [f"{counts[n][part]:,}" for n in names]
        parts.add_row(part, *cells, style="bold" if part == "total" else "")
    # float32: 4 bytes per parameter. The LM head adds nothing: it reuses the token embeddings.
    parts.add_row("MB (fp32)", *(f"{counts[n]['total'] * 4 / 2**20:.1f}" for n in names))
    console.print(parts)


def tokenizer_train(args: argparse.Namespace) -> None:
    """Train the BPE tokenizer on web text plus weighted schemas and commands."""
    from eve.data.corpus import iter_documents, take_bytes, tokenizer_texts
    from eve.tokenizer.bpe import BPETokenizer

    corpus = take_bytes(iter_documents(_parquet_files(args.parquet)), _count(args.text))
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
        train_tokens=_count(args.train_tokens),
        val_tokens=_count(args.val_tokens),
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


def corpus_prompts(args: argparse.Namespace) -> None:
    """Write the prompts of a TinyTools run: one document to write per line."""
    from eve.data.generate.tinytools import KINDS, Lexicon, make_prompts
    from eve.data.jsonl import write_jsonl

    path = Path(args.out) / "prompts.jsonl"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; documents may already follow it (--force replaces it)")
    specs = make_prompts(Lexicon.load(args.lexicon), args.n, args.seed)
    write_jsonl(path, (spec.to_json() for spec in specs))
    counts = Counter(spec.kind for spec in specs)
    kinds = " · ".join(f"{kind} {counts[kind]:,}" for kind in KINDS)
    domains = len({spec.domain for spec in specs})
    success(f"{len(specs):,} prompts ({kinds}) over {domains} domains", path)


def corpus_generate(args: argparse.Namespace) -> None:
    """Write a TinyTools run's documents with headless Claude Code, then check the run."""
    from eve.data.generate.headless import (
        BatchResult,
        GenerateError,
        claude_writer,
        generate,
        pending,
        read_documents,
    )
    from eve.data.generate.tinytools import Lexicon, PromptSpec
    from eve.data.jsonl import read_jsonl
    from eve.data.schema import load_library
    from eve.progress import Bar

    run = Path(args.dir)
    if not (run / "prompts.jsonl").is_file():
        raise SystemExit(f"no {run / 'prompts.jsonl'}; write it with eve corpus prompts")
    specs = [PromptSpec.from_json(row) for row in read_jsonl(run / "prompts.jsonl")]
    lexicon = Lexicon.load(args.lexicon)
    heldout = list(load_library(Path(args.tools) / "heldout").values())
    documents = read_documents(run / "documents.jsonl")
    todo, previous = pending(specs, documents, lexicon, heldout, args.rewrite)
    if args.limit is not None:
        todo = todo[: args.limit]
    batches = -(-len(todo) // args.batch)
    console.print(
        panel(
            "eve corpus generate",
            [
                ("run", str(run)),
                ("written", f"{len(documents):,} of {len(specs):,} prompts"),
                (
                    "to write",
                    f"{len(todo):,} ({len(previous):,} rewrites)" if previous else f"{len(todo):,}",
                ),
                (
                    "batches",
                    f"{batches:,} of {args.batch} · {args.workers} at a time · "
                    f"{args.rounds} rounds",
                ),
                ("generator", f"{args.claude} -p · model {args.model or 'default'}"),
            ],
        )
    )
    if todo:
        cost, accepted = 0.0, 0
        with Bar(len(todo), "writing", "documents") as progress:

            def on_batch(result: BatchResult) -> None:
                nonlocal cost, accepted
                cost, accepted = cost + result.cost, accepted + result.accepted
                progress.update(result.size)
                progress.set_stats(f"{accepted:,} accepted", f"${cost:.2f}")
                line = (
                    f"batch {result.number}: {result.accepted} accepted, "
                    f"{len(result.rejected)} rejected, {len(result.documents)} written · "
                    f"{result.calls} calls · ${result.cost:.2f}"
                )
                if result.errors:
                    line += f" · {len(result.errors)} errors: {result.errors[-1]}"
                if progress.disable or result.errors:
                    progress.write(line)

            write = claude_writer(args.claude, args.model, args.timeout)
            try:
                generate(
                    run,
                    todo,
                    write,
                    lexicon,
                    heldout,
                    previous,
                    args.batch,
                    args.workers,
                    args.rounds,
                    on_batch,
                )
            except GenerateError as error:
                raise SystemExit(f"stopped: {error}; run the command again to resume") from None
        console.print(f"wrote {accepted:,} accepted documents for ${cost:.2f}")
    corpus_check(args)


def corpus_check(args: argparse.Namespace) -> None:
    """Check a TinyTools run's documents against their prompts; write the accepted ones."""
    from eve.data.generate.tinytools import (
        Lexicon,
        PromptSpec,
        check_corpus,
        write_parquet,
    )
    from eve.data.jsonl import read_jsonl
    from eve.data.schema import load_library

    run = Path(args.dir)  # not args.run: that is the command's function
    prompts, documents = run / "prompts.jsonl", run / "documents.jsonl"
    for path in (prompts, documents):
        if not path.is_file():
            raise SystemExit(f"no {path}; see data/tinytools/README.md")
    specs = [PromptSpec.from_json(row) for row in read_jsonl(prompts)]
    texts = {str(row["id"]): str(row["text"]) for row in read_jsonl(documents)}
    lexicon = Lexicon.load(args.lexicon)
    heldout = list(load_library(Path(args.tools) / "heldout").values())
    report = check_corpus(specs, texts, lexicon, heldout)
    stats = report.stats(lexicon)
    write_parquet(report.accepted, run / "tinytools.parquet")
    (run / "report.json").write_text(
        json.dumps(
            {"stats": stats, "rejected": report.rejected, "missing": report.missing}, indent=1
        )
    )
    if args.json:
        print(json.dumps(stats))
        return
    _print_corpus_report(stats, report.rejected, run)


def _print_corpus_report(stats: dict[str, Any], rejected: dict[str, list[str]], run: Path) -> None:
    """The counts, the most common reject reasons and a few rejected documents."""
    kinds = " · ".join(f"{k} {n:,}" for k, n in stats["accepted_by_kind"].items())
    items: list[tuple[str, str | Text]] = [
        ("accepted", f"{stats['accepted']:,} of {stats['prompts']:,} prompts · {kinds}"),
        ("rejected", f"{stats['rejected']:,}"),
        ("not written", f"{stats['missing']:,}"),
        ("words", f"{stats['words']:,} · {stats['mean_words']:.0f} per document"),
        (
            "covered",
            f"{stats['vocabulary_covered']:.0%} of the words · {stats['domains_covered']} domains",
        ),
    ]
    if stats["unknown"]:
        items.append(("unknown ids", f"{stats['unknown']:,} documents match no prompt"))
    for i, (reason, n) in enumerate(stats["reject_reasons"].items()):
        if i == 5:
            break
        items.append(("reasons" if i == 0 else "", f"{n:,} × {reason}"))
    for i, (doc_id, reasons) in enumerate(rejected.items()):
        if i == 3:
            break
        items.append(("examples" if i == 0 else "", f"{doc_id}: {'; '.join(reasons)}"))
    phrases = [
        f'{kind}: "{phrase}" {share:.0%}'
        for kind, top in stats["repeated_phrases"].items()
        for phrase, share in list(top.items())[:1]
    ]
    for i, line in enumerate(phrases):
        items.append(("top phrase" if i == 0 else "", line))
    copied = [(f, share) for f, share in stats["copied_examples"].items() if share > 0.3]
    for i, (feature, share) in enumerate(copied):
        items.append(("copied" if i == 0 else "", f"{share:.0%} of {feature}"))
    console.print(panel("eve corpus check", items))
    success("accepted documents", run / "tinytools.parquet")


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


def _human(count: float) -> str:
    """250_000_000 -> '250M', 2_300_000 -> '2.3M', 900 -> '900'."""
    for scale, suffix in ((10**9, "B"), (10**6, "M"), (10**3, "k")):
        if count >= scale:
            return f"{count / scale:.3g}{suffix}"
    return str(round(count))


def _schedule(args: argparse.Namespace, lr: float, steps: int) -> str:
    """The panel row both training runs share: steps, batch size, lr and warmup."""
    return f"{steps:,} steps × {args.batch_size} · lr {lr:g} · warmup {min(args.warmup, steps):,}"


def _run_rows(args: argparse.Namespace, device: torch.device) -> list[tuple[str, str]]:
    """The panel rows both training runs share: where it runs and where it writes."""
    hardware = [str(device), args.dtype] + (["compiled"] if getattr(args, "compile", False) else [])
    run = [args.out, f"eval every {args.eval_every:,}"]
    run += [f"stop at {args.stop_after:,}"] if args.stop_after else []
    run += ["resume"] if args.resume else []
    config = [("config", args.config)] if args.config else []
    config += [("set", " · ".join(args.set))] if args.set else []
    return [*config, ("device", " · ".join(hardware)), ("output", " · ".join(run))]


def _pretrain_panel(
    args: argparse.Namespace, cfg: ModelConfig, device: torch.device, steps: int
) -> "Panel":
    """What ``pretrain`` is about to do: the model, data, schedule, device and output."""
    from eve.data.pretrain import raw_length

    options = [name for name in ("rope", "qk_norm") if getattr(cfg, name)]
    model = [args.preset, f"{cfg.approx_params() / 1e6:.1f}M params", *options]
    tokens = steps * args.batch_size * raw_length(cfg.max_encoder_len)
    items = [
        ("model", " · ".join([*model, f"dropout {cfg.dropout:g}"])),
        ("data", f"{args.data} · {_human(tokens)} tokens in windows of {cfg.max_encoder_len}"),
        ("schedule", _schedule(args, args.lr, steps)),
    ]
    return panel("eve pretrain", items + _run_rows(args, device))


def _run_experiment(values: dict[str, Any], sets: list[str]) -> None:
    """Apply the ``--set`` overrides to an experiment's sections and run it."""
    from eve.experiment import Experiment, apply_sets, run

    try:
        experiment = Experiment.from_dict(apply_sets(values, sets))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"bad experiment: {error}") from error
    run(experiment)


def pretrain(args: argparse.Namespace) -> None:
    """Run span-corruption pretraining with the command line's options."""
    from eve.config import preset
    from eve.train.pretrain import pretrain_experiment, steps_for

    overrides = {"rope": args.rope, "qk_norm": args.qk_norm}
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    cfg = preset(args.preset, **overrides)
    steps = args.steps or steps_for(_count(args.tokens), args.batch_size, cfg.max_encoder_len)
    train_cfg = _train_config(args, steps, args.lr, compile=args.compile)
    device = pick_device(args.device)
    console.print(_pretrain_panel(args, cfg, device, steps))
    values = pretrain_experiment(
        {"preset": args.preset, **overrides},
        train_cfg,
        args.data,
        args.tokenizer,
        args.out,
        device,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    _run_experiment(values, args.set)


def _finetune_panel(
    args: argparse.Namespace, device: torch.device, lr: float, steps: int, n_train: int
) -> "Panel":
    """What ``finetune`` is about to do: where from, on what, how long, and where to."""
    start = [args.init or f"{args.scratch} preset, random weights"]
    start += ["body frozen"] if args.freeze else []
    epochs = steps * args.batch_size / n_train
    items = [
        ("start", " · ".join(start)),
        ("task", f"{args.task} · {n_train:,} train examples"),
        ("schedule", f"{_schedule(args, lr, steps)} · {epochs:.1f} epochs"),
    ]
    return panel("eve finetune", items + _run_rows(args, device))


def finetune(args: argparse.Namespace) -> None:
    """Fine-tune the question heads and argument values with the command line's options."""
    from eve.data.jsonl import read_jsonl
    from eve.train.finetune import finetune_experiment

    if (args.init is None) == (args.scratch is None):
        raise SystemExit("give one of --init or --scratch (on the command line or in --config)")
    n_train = len(read_jsonl(Path(args.task) / "train.jsonl"))
    steps = args.steps or round(args.epochs * n_train / args.batch_size)
    lr = args.lr or (FROZEN_LR if args.freeze else FULL_LR)
    train_cfg = _train_config(args, steps, lr, eval_batches=ALL_VAL_BATCHES)
    device = pick_device(args.device)
    console.print(_finetune_panel(args, device, lr, steps, n_train))
    values = finetune_experiment(
        train_cfg,
        args.task,
        args.out,
        device,
        init=args.init,
        scratch=args.scratch,
        tokenizer_path=args.tokenizer,
        freeze=args.freeze,
        overrides={} if args.dropout is None else {"dropout": args.dropout},
        resume=args.resume,
        stop_after=args.stop_after,
    )
    _run_experiment(values, args.set)


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
        records = [json.loads(line) for line in path.read_text().splitlines()]
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
    """Learn (M4): val metrics of several runs side by side, one row per eval step."""
    console.print(_compare_table(_read_runs(args.runs), args.metric))


def _print_fill_report(
    model_path: str, meta: dict[str, Any], device: torch.device, filled: list[str]
) -> None:
    """Which checkpoint filled the holes and how far it was trained, then each filled text."""
    from eve.tasks.span import HIGHLIGHTS

    trained = f"{meta['step']:,} steps" if "step" in meta else "?"
    if "val_lm" in meta:
        trained += f" · val loss {meta['val_lm']:.3f}"
    about = [("checkpoint", model_path), ("trained", trained), ("device", str(device))]
    console.print(panel("eve fill", about))
    texts = [Text(line) for line in filled]
    for text in texts:
        for pattern, style in HIGHLIGHTS:
            text.highlight_regex(pattern, style)
    console.print(rows([(str(i), text) for i, text in enumerate(texts, 1)], indent=2))
    console.print()


def fill(args: argparse.Namespace) -> None:
    """Learn (M4): ask a pretrained model to fill the <x0>, <x1>… holes in a sentence."""
    from eve import progress
    from eve.model.checkpoint import load_checkpoint
    from eve.tasks.span import SAMPLES, fill_in

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
            filled.append(fill_in(ckpt.model, ckpt.tokenizer, text, args.max_new))
            bar.update()

    _print_fill_report(str(args.model), ckpt.meta, device, filled)


def _tools(args: argparse.Namespace) -> list["Tool"]:
    """The tools for ``eve ask``: names from the library and/or schemas from JSON files."""
    from eve.data.generate.build import Sources
    from eve.data.schema import GENERATION_KEYS, Tool

    tools = []
    if args.tools:
        library = Sources.load().library
        names = [name.strip() for name in args.tools.split(",") if name.strip()]
        if unknown := [name for name in names if name not in library]:
            raise SystemExit(f"unknown tools {unknown}; see data/tools/*.json or use --schema")
        tools += [library[name] for name in names]
    for path in args.schema or []:
        data = json.loads(Path(path).read_text())
        schemas = data["tools"] if isinstance(data, dict) else data  # a data/tools file, or a list
        tools += [
            Tool.from_dict({k: v for k, v in t.items() if k not in GENERATION_KEYS})
            for t in schemas
        ]
    if not tools:
        raise SystemExit("give the tools with --tools a,b,c and/or --schema FILE")
    return tools


def _confidence(prob: float) -> str:
    """The color of an answer this likely: green when sure, yellow when unsure, red below."""
    return "green" if prob >= SURE else "yellow" if prob >= UNSURE else "red"


def _ranked(probs: dict[str, float], count: int) -> Text:
    """The ``count`` most likely answers with their probabilities: the first in bold, colored
    by how likely it is, the others dimmed."""
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:count]
    return Text(" · ", style="dim").join(
        Text(f"{k} {p:.0%}", style=f"bold {_confidence(p)}" if i == 0 else "dim")
        for i, (k, p) in enumerate(top)
    )


def _values(call: "WrittenCall") -> Text:
    """``room kitchen 98% · on false 99% · brightness –``: each argument and what the decoder
    wrote, colored by how likely it is; ``?`` for one to ask the user for, ``✗`` if invalid."""
    parts = []
    for name, value in call.values.items():
        part = Text(f"{name} ")
        if value.kind == "absent":
            part = Text(f"{name} –", style="dim")
        elif value.kind == "ask":
            part.append("?", style="bold yellow")
        elif value.kind == "invalid":
            part.append(f"{value.value!r} ✗", style="bold red")
        else:
            shown = value.value if isinstance(value.value, str) else json.dumps(value.value)
            part.append(str(shown), style=f"bold {_confidence(value.prob)}")
        if value.kind != "absent":
            part.append(f" {value.prob:.0%}", style="dim")
        parts.append(part)
    return Text(" · ", style="dim").join(parts) if parts else Text("no arguments", style="dim")


def _print_answers(text: str, answers: "Answers", as_json: bool) -> None:
    """The heads' answers for one request: one JSON line, or the intent and each call slot up
    to the first one that says STOP, with its argument values if they were written."""
    if as_json:
        print(json.dumps({"utterance": text, **answers.to_json()}))
        return
    items = [("intent", _ranked(answers.intent, INTENTS_SHOWN))]
    for i, slot in enumerate(answers.slots, 1):
        items.append((f"call {i}", _ranked(slot, TOOLS_SHOWN)))
        if i <= len(answers.written):
            items.append(("", _values(answers.written[i - 1])))
        if max(slot, key=slot.__getitem__) == STOP:
            break
    console.print(rows(items, indent=2))
    console.print()


def _ask_interactively(show: Callable[[str], None]) -> None:
    """Read requests from the terminal and show the answers, until an empty line or Ctrl-D."""
    while True:
        try:
            text = console.input(Text("› ", style="bold cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            break
        show(text)


def ask(args: argparse.Namespace) -> None:
    """Learn (M7, M8): what the question heads answer for a typed request, with probabilities,
    and the argument values of each call, for a model from the joint run. The probabilities
    use the checkpoint's temperatures if ``eve calibrate --save`` wrote them (M9): the
    familiar-tool ones if the model trained on every tool given, else the new-tool ones."""
    from eve.infer.predict import Calibrations, answer_questions
    from eve.model.checkpoint import load_checkpoint

    tools = _tools(args)
    ckpt = load_checkpoint(args.model, pick_device(args.device))
    values = bool(ckpt.meta.get("joint"))  # earlier runs never trained the decoder to write them
    calibrations = Calibrations.from_meta(ckpt.meta)
    calibration = calibrations.for_tools(tools)

    def show(text: str) -> None:
        answers = answer_questions(ckpt.model, ckpt.tokenizer, text, tools, values, calibration)
        _print_answers(text, answers, args.json)

    if args.text:
        for text in args.text:
            if not args.json:
                console.print(Text(text, style="bold"))
            show(text)
        return
    names = ", ".join(t.name for t in tools)
    written = "written" if values else "not written: the model is not from a joint run"
    regime = "familiar" if calibrations.is_familiar(tools) else "new"
    fitted = ckpt.meta.get("calibrated_on", {}).get(regime)
    probs = f"calibrated for {regime} tools" if fitted else "as trained: see eve calibrate"
    items = [("model", args.model), ("tools", names), ("values", written), ("probs", probs)]
    console.print(panel("eve ask", items))
    console.print(Text("type a request · empty line or Ctrl-D to quit\n", style="dim"))
    _ask_interactively(show)


def _evaluate_panel(
    args: argparse.Namespace, meta: dict[str, Any], device: torch.device
) -> "Panel":
    """Which model is scored, how it was trained, and on what."""
    from eve.evaluate import model_file

    trained = [f"{meta['step']:,} steps" if "step" in meta else "? steps", f"from {meta['init']}"]
    trained += ["body frozen"] if meta.get("frozen") else []
    items = [
        ("model", str(model_file(args.model))),
        ("trained", " ".join(trained)),
        ("task", f"{args.task} · {', '.join(args.sets)}"),
        ("device", str(device)),
    ]
    return panel("eve evaluate", items)


def _score_cell(scores: dict[str, float], key: str) -> str:
    """A recall per intent as ``96%``, the summary scores as ``53.3%``, as in the training log.
    Blank when the set has no examples of that intent."""
    if key not in scores:
        return ""
    return f"{scores[key]:.0%}" if key.startswith("recall_") else f"{scores[key]:.1%}"


def _scores_table(results: dict[str, "SetResult"]) -> "Table":
    """One row per scored set: its size, the recall per intent and their mean, tools and exact.
    A set without examples of an intent has a blank recall for it."""
    out = table("set", "n", *(label for label, _ in SCORE_COLUMNS), first_style="bold")
    for name, result in results.items():
        cells = [_score_cell(result.scores, key) for _, key in SCORE_COLUMNS]
        out.add_row(name, f"{result.examples:,}", *cells)
    return out


def _confusion_table(name: str, result: "SetResult") -> "Table":
    """A row per gold intent the set has, a column per intent answered; right answers green."""
    out = table(f"{name}: gold \\ answered", *INTENTS, first_style="dim")
    for gold, said in result.confusion.items():
        if sum(said.values()):
            styles = ("bold green" if s == gold else "dim" * (not n) for s, n in said.items())
            out.add_row(
                gold,
                *(Text(f"{n:,}", style) for n, style in zip(said.values(), styles, strict=True)),
            )
    return out


def _answered(counts: dict[str, int]) -> Text:
    """``call 12 · unsup 3``: how often each intent was answered, most often first."""
    ranked = sorted(((n, intent) for intent, n in counts.items() if n), reverse=True)
    return Text(" · ").join(Text(f"{intent[:5]} {n:,}") for n, intent in ranked)


def _tools_table(name: str, result: "SetResult") -> "Table":
    """A row per gold tool, worst exact accuracy first, with what its examples were answered."""
    out = table("gold tool", "n", "intent", "tools", "exact", "values", "full", "answered")
    out.title, out.title_justify, out.title_style = f"{name} by tool", "left", "bold"
    out.columns[-1].justify, out.columns[-1].no_wrap = "left", False
    for tool in result.by_tool:
        accuracies = (
            tool.intent_acc,
            tool.tools_acc,
            tool.exact_acc,
            tool.value_acc,
            tool.full_acc,
        )
        cells = [f"{tool.examples:,}", *("" if a is None else f"{a:.0%}" for a in accuracies)]
        out.add_row(tool.tool, *cells, _answered(tool.answered))
    return out


def _print_evaluation(results: dict[str, "SetResult"], as_json: bool, by_tool: bool) -> None:
    """The scores table and a confusion table per set (and a per-tool table with ``by_tool``),
    or all of it as one JSON object."""
    if as_json:
        print(json.dumps({name: result.to_json(by_tool) for name, result in results.items()}))
        return
    console.print(_scores_table(results))
    for name, result in results.items():
        console.print(_confusion_table(name, result))
        if by_tool:
            console.print(_tools_table(name, result))


def evaluate(args: argparse.Namespace) -> None:
    """Score a fine-tuned model on whole task files: recall per intent, tools and exact."""
    from eve import progress
    from eve.data.generate.build import Sources
    from eve.evaluate import evaluate_set, load_questions_model, task_files

    device = pick_device(args.device)
    try:
        paths = task_files(args.task, args.sets)
        ckpt = load_questions_model(args.model, device)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not args.json:
        console.print(_evaluate_panel(args, ckpt.meta, device))
    library, results = Sources.load().library, {}
    with progress.bar(len(paths), "evaluate", "sets", leave=False) as bar:
        for name, path in paths.items():
            bar.set_stats(name)
            results[name] = evaluate_set(ckpt, path, library, args.batch_size)
            bar.update()
    _print_evaluation(results, args.json, args.by_tool)


def _calibration_table(
    name: str, counts: dict[str, int], before: dict[str, Any], after: dict[str, Any]
) -> "Table":
    """A row per answer: how many, their accuracy, and the mean confidence and expected
    calibration error before and after (``eve.calibrate.Reliability``)."""
    out = table(name, "n", "accuracy", "confidence", "", "ECE", "", first_style="bold")
    out.title = f"{counts['familiar']:,} familiar · {counts['new']:,} new"
    out.title_justify, out.title_style = "left", "dim"
    for head, old in before.items():
        new = after[head]
        out.add_row(
            head,
            f"{old.n:,}",
            f"{old.accuracy:.1%}",
            Text(f"{old.mean_confidence:.1%} →", style="dim"),
            f"{new.mean_confidence:.1%}",
            Text(f"{old.ece():.3f} →", style="dim"),
            Text(f"{new.ece():.3f}", style="bold green" if new.ece() < old.ece() else "bold red"),
        )
    return out


def calibrate(args: argparse.Namespace) -> None:
    """Fit temperatures for familiar and for new tools (M9) and show how calibrated the
    probabilities are before and after on each set; ``--save`` writes them into the model."""
    from eve import progress
    from eve.calibrate import (
        HEADS,
        REGIMES,
        fit,
        known_tools,
        load_set,
        reliability,
        save_calibration,
    )
    from eve.data.generate.build import Sources
    from eve.evaluate import load_questions_model, model_file, task_files
    from eve.infer.predict import Calibrations

    device = pick_device(args.device)
    fit_on = {"familiar": args.fit_familiar, "new": args.fit_new}
    sets = list(dict.fromkeys([*fit_on.values(), *args.sets]))  # the fit sets first, once
    try:
        paths = task_files(args.task, [*sets, "train"])
        ckpt = load_questions_model(args.model, device)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not ckpt.meta.get("joint"):
        raise SystemExit(f"{args.model} is not from a joint run: its decoder writes no values")
    library = Sources.load().library
    with progress.bar(len(REGIMES) + len(sets), "calibrate", "passes", leave=False) as bar:
        bar.set_stats("reading the training tools")
        known = Calibrations(known=known_tools(paths["train"], library))
        tasks = {
            name: load_set(ckpt, paths[name], library, args.batch_size, known) for name in sets
        }
        fitted = {}
        for regime, name in fit_on.items():
            bar.set_stats(f"fitting {regime} on {name}")
            fitted[regime] = fit(ckpt.model, tasks[name].batches, tasks[name].only(regime))
            bar.update()
        cal = Calibrations(fitted["familiar"], fitted["new"], known.known)
        results = {}
        for name in sets:
            bar.set_stats(name)
            results[name] = reliability(ckpt.model, tasks[name], [known, cal])
            bar.update()
    if args.save:
        save_calibration(model_file(args.model), cal, fit_on)

    if args.json:
        sets_json = {
            name: {
                "examples": tasks[name].counts(),
                **{h: {"before": old[h].to_json(), "after": new[h].to_json()} for h in HEADS},
            }
            for name, (old, new) in results.items()
        }
        temperatures = {"familiar": cal.familiar.to_dict(), "new": cal.new.to_dict()}
        out = {"calibration": temperatures, "fitted_on": fit_on, "sets": sets_json}
        print(json.dumps(out))
        return
    items: list[tuple[str, str | Text]] = [("model", str(args.model))]
    for regime in REGIMES:
        temps = getattr(cal, regime)
        line = " · ".join(f"{head} {getattr(temps, head):.2f}" for head in HEADS)
        items.append((regime, f"{line}   fitted on {fit_on[regime]}"))
    items.append(("known tools", f"{len(cal.known):,} schemas in {paths['train']}"))
    console.print(panel("eve calibrate", items))
    for name, (old, new) in results.items():
        console.print(_calibration_table(name, tasks[name].counts(), old, new))
    if args.save:
        success("temperatures saved", model_file(args.model))
    else:
        console.print(Text("not saved: add --save to write them into the model", style="dim"))


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
        args.templates,
        args.heldout,
        args.seed,
        tokenizer,
        preset("small"),
    )
    console.print(_splits_table(report["splits"]))
    if rejected := report["rejected_chat_lines"]:
        warn(f"{rejected} chat-written lines rejected (see `eve data check`)")
    success("built", Path(args.out) / "report.json")


def _splits_table(splits: dict[str, dict[str, Any]]) -> "Table":
    """Examples per split, in total and per intent."""
    from eve.config import INTENTS

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


def _add_schedule_args(
    parser: argparse.ArgumentParser, lr: float | None, lr_help: str | None, warmup: int
) -> None:
    """Batch size, learning rate, warmup and precision: the options both training runs take."""
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=lr, help=lr_help)
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=TRAIN_DEFAULTS.dtype)


def _add_switch(parser: argparse.ArgumentParser, flag: str, help_: str) -> None:
    """An off-by-default ``--flag``, with a ``--no-flag`` to turn off what a config turned on."""
    parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=False, help=help_)


def _add_config_arg(parser: argparse.ArgumentParser, command: str) -> None:
    """``--config NAME``: the options in ``configs/<command>/NAME.toml`` become the defaults."""
    names = ", ".join(available(command)) or "none yet"
    parser.add_argument(
        "--config",
        default=None,
        help=f"a configs/{command}/*.toml name or path ({names}); flags still win",
    )
    parser.set_defaults(parser=parser)


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
    _add_set_arg(parser)


def _add_set_arg(parser: argparse.ArgumentParser) -> None:
    """``--set section.key=value``: change any experiment value, e.g. components.loss=question."""
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override an experiment value (repeatable), e.g. components.loss=question",
    )


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
    group = commands.add_parser(
        "corpus", help="the pretraining corpus: encode it, or build TinyTools"
    )
    corpus_commands = group.add_subparsers(dest="corpus_command", required=True)
    parser = corpus_commands.add_parser("encode", help="encode parquet text to token ids (M1)")
    parser.set_defaults(run=corpus_encode)
    parser.add_argument("--parquet", default=defaults.PARQUET_GLOB)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
    parser.add_argument("--out", default=defaults.BIN_DIR)
    parser.add_argument("--train-tokens", default="250M")
    parser.add_argument("--val-tokens", default="2M")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--mix", help="a glob of parquet files to spread through the train tokens, e.g. TinyTools"
    )
    parser.add_argument(
        "--mix-share", type=float, default=0.3, help="their share of the train text (with --mix)"
    )
    parser.add_argument("--seed", type=int, default=0, help="the order of the mixed documents")

    parser = corpus_commands.add_parser("prompts", help="TinyTools: prompts for a generator")
    parser.set_defaults(run=corpus_prompts)
    parser.add_argument("--n", type=int, default=2000, help="documents to write")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=defaults.TINYTOOLS_RUN, help="the run directory")
    parser.add_argument("--lexicon", default=defaults.TINYTOOLS_DIR, help="word lists and domains")
    parser.add_argument("--force", action="store_true", help="replace an existing prompts.jsonl")

    parser = corpus_commands.add_parser(
        "generate", help="TinyTools: write the documents with headless Claude Code"
    )
    parser.set_defaults(run=corpus_generate, json=False)
    parser.add_argument(
        "dir", nargs="?", default=defaults.TINYTOOLS_RUN, metavar="run", help="the run directory"
    )
    parser.add_argument("--batch", type=int, default=50, help="prompts per call")
    parser.add_argument("--workers", type=int, default=4, help="calls at a time")
    parser.add_argument(
        "--rounds", type=int, default=3, help="calls per batch: the first, then rewrites"
    )
    parser.add_argument("--limit", type=int, default=None, help="write at most this many")
    parser.add_argument(
        "--rewrite", action="store_true", help="also rewrite documents the checker rejects"
    )
    parser.add_argument("--model", default=None, help="e.g. sonnet; Claude Code's default if unset")
    parser.add_argument("--claude", default="claude", help="the Claude Code command")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per call")
    parser.add_argument("--lexicon", default=defaults.TINYTOOLS_DIR)
    parser.add_argument("--tools", default=defaults.TOOLS_DIR, help="for the held-out tools")

    parser = corpus_commands.add_parser("check", help="TinyTools: check written documents")
    parser.set_defaults(run=corpus_check)
    parser.add_argument(
        "dir", nargs="?", default=defaults.TINYTOOLS_RUN, metavar="run", help="the run directory"
    )
    parser.add_argument("--lexicon", default=defaults.TINYTOOLS_DIR)
    parser.add_argument("--tools", default=defaults.TOOLS_DIR, help="for the held-out tools")
    parser.add_argument("--json", action="store_true", help="the stats as one JSON line")


def _add_pretrain_command(commands: Subcommands) -> None:
    parser = commands.add_parser("pretrain", help="span-corruption pretraining (M4)")
    parser.set_defaults(run=pretrain)
    _add_config_arg(parser, "pretrain")
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
    _add_switch(parser, "--rope", "rotary positions, no position tables")
    _add_switch(parser, "--qk-norm", "RMS-normalize queries and keys")
    _add_switch(parser, "--compile", "torch.compile: ~2x faster on MPS")
    _add_run_args(parser)


def _add_finetune_command(commands: Subcommands) -> None:
    parser = commands.add_parser("finetune", help="fine-tune questions and values (M7, M8)")
    parser.set_defaults(run=finetune)
    _add_config_arg(parser, "finetune")
    start = parser.add_mutually_exclusive_group()  # one is required; finetune checks
    start.add_argument("--init", help="a pretraining checkpoint or an earlier finetune's model.pt")
    start.add_argument("--scratch", choices=sorted(PRESETS), help="random init (experiment E1)")
    _add_switch(parser, "--freeze", "train only the question block (E2)")
    parser.add_argument("--task", default=defaults.TASK_DIR)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE, help="for --scratch")
    parser.add_argument("--out", default=defaults.FINETUNE_RUN)
    parser.add_argument("--epochs", type=float, default=2.0, help="sets --steps")
    parser.add_argument("--steps", type=int, default=None, help="default: from --epochs")
    _add_schedule_args(
        parser,
        lr=None,
        lr_help=f"default: {FROZEN_LR:g} frozen, else {FULL_LR:g}",
        warmup=FINETUNE_WARMUP,
    )
    parser.add_argument("--dropout", type=float, default=None, help="default: the checkpoint's")
    _add_run_args(parser)


def _add_run_command(commands: Subcommands) -> None:
    parser = commands.add_parser("run", help="run an experiment file, e.g. a run's config.toml")
    parser.set_defaults(run=run_file)
    parser.add_argument("experiment", help="configs/experiments/*.toml, or <run>/config.toml")
    _add_set_arg(parser)


def _add_inspect_commands(commands: Subcommands) -> None:
    parser = commands.add_parser("ask", help="ask the model about a request (M7, M8)")
    parser.set_defaults(run=ask)
    parser.add_argument("text", nargs="*", help="requests; none: type them one by one")
    parser.add_argument("--tools", help="tool names from data/tools: set_lights,set_thermostat")
    parser.add_argument(
        "--schema", action="append", help="a JSON file of tool schemas (repeatable)"
    )
    parser.add_argument("--model", default=defaults.FINETUNED_MODEL)
    parser.add_argument("--json", action="store_true", help="one JSON line per request")
    parser.add_argument("--device", default=None)

    _add_evaluate_command(commands)
    _add_calibrate_command(commands)
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


def _add_evaluate_command(commands: Subcommands) -> None:
    from eve.evaluate import SETS

    parser = commands.add_parser("evaluate", help="score a fine-tuned model on whole task files")
    parser.set_defaults(run=evaluate)
    parser.add_argument(
        "model", nargs="?", default=defaults.FINETUNED_MODEL, help="a model.pt or a run directory"
    )
    parser.add_argument("--task", default=defaults.TASK_DIR)
    parser.add_argument(
        "--sets", nargs="+", default=list(SETS), help="task file names, e.g. probe-candidates"
    )
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--by-tool", action="store_true", help="also a row per gold tool")
    parser.add_argument("--json", action="store_true", help="the results as one JSON object")
    parser.add_argument("--device", default=None)


def _add_calibrate_command(commands: Subcommands) -> None:
    from eve.evaluate import SETS

    parser = commands.add_parser("calibrate", help="fit the probabilities' temperatures (M9)")
    parser.set_defaults(run=calibrate)
    parser.add_argument(
        "model", nargs="?", default=defaults.FINETUNED_MODEL, help="a model.pt or a run directory"
    )
    parser.add_argument("--task", default=defaults.TASK_DIR)
    parser.add_argument(
        "--fit-familiar", default="val", help="task file whose familiar-tool examples fit those"
    )
    parser.add_argument(
        "--fit-new", default="probe-candidates", help="task file whose new-tool examples fit those"
    )
    parser.add_argument("--sets", nargs="+", default=list(SETS), help="task files to check them on")
    parser.add_argument("--save", action="store_true", help="write them into the model file")
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--json", action="store_true", help="the results as one JSON object")
    parser.add_argument("--device", default=None)


def _add_data_build_command(data_commands: Subcommands) -> None:
    parser = data_commands.add_parser("build", help="templates + chat-written -> data/task/*.jsonl")
    parser.set_defaults(run=data_build)
    parser.add_argument("--out", default=defaults.TASK_DIR)
    parser.add_argument(
        "--templates",
        type=_count,
        default=defaults.TEMPLATE_EXAMPLES,
        help="template examples for train",
    )
    parser.add_argument(
        "--heldout",
        type=_count,
        default=defaults.HELDOUT_EXAMPLES,
        help="template examples with held-out tools",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)


def _add_data_commands(commands: Subcommands) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    """The ``eve`` command line: one subcommand per stage, each with a ``run`` function."""
    parser = argparse.ArgumentParser(prog="eve")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_model_commands(commands)
    _add_tokenizer_commands(commands)
    _add_corpus_command(commands)
    _add_pretrain_command(commands)
    _add_finetune_command(commands)
    _add_run_command(commands)
    _add_inspect_commands(commands)
    _add_data_commands(commands)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line's options; with ``--config``, the config file fills in the defaults.

    The command line is parsed twice: once to find the config, and again after the config's
    values became the subcommand's defaults, so a flag given on the command line still wins.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "config", None) is None:
        return args
    try:
        options = read_config(config_path(args.command, args.config))
    except (FileNotFoundError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"{args.config}: {error}") from error
    apply_config(args.parser, options)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the chosen subcommand."""
    args = parse_args(argv)
    setup_logging()
    args.run(args)
