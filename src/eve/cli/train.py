"""Commands that train and compare runs: ``pretrain``, ``finetune``, ``run`` and ``compare``."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rich.text import Text

from eve import defaults
from eve.cli.common import TRAIN_DEFAULTS, Subcommands, add_device_arg, count
from eve.config import PRESETS, ModelConfig
from eve.progress import console, format_metric, higher_is_better, panel, table
from eve.runtime import pick_device
from eve.train.config import DTYPES, TrainConfig

if TYPE_CHECKING:
    from rich.panel import Panel
    from rich.table import Table


DTYPE_CHOICES = list(DTYPES)


FINETUNE_LR = 3e-4


ALL_VAL_BATCHES = 10_000  # finetune evaluates on all of val: it is small


FINETUNE_WARMUP = 200


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
    steps = args.steps or steps_for(count(args.tokens), args.batch_size, cfg.max_encoder_len)
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
    epochs = steps * args.batch_size / n_train
    items = [
        ("start", args.init or f"{args.scratch} preset, random weights"),
        ("task", f"{args.task} · {n_train:,} train examples"),
        ("schedule", f"{_schedule(args, lr, steps)} · {epochs:.1f} epochs"),
    ]
    return panel("eve finetune", items + _run_rows(args, device))


def finetune(args: argparse.Namespace) -> None:
    """Fine-tune the question heads and argument values with the command line's options."""
    from eve.data.jsonl import read_jsonl
    from eve.train.finetune import finetune_experiment

    if (args.init is None) == (args.scratch is None):
        raise SystemExit("give one of --init or --scratch")
    n_train = len(read_jsonl(Path(args.task) / "train.jsonl"))
    steps = args.steps or round(args.epochs * n_train / args.batch_size)
    train_cfg = _train_config(args, steps, args.lr, eval_batches=ALL_VAL_BATCHES)
    device = pick_device(args.device)
    console.print(_finetune_panel(args, device, args.lr, steps, n_train))
    values = finetune_experiment(
        train_cfg,
        args.task,
        args.out,
        device,
        init=args.init,
        scratch=args.scratch,
        tokenizer_path=args.tokenizer,
        overrides={} if args.dropout is None else {"dropout": args.dropout},
        resume=args.resume,
        stop_after=args.stop_after,
        evals_dir=args.evals,
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
    """Val metrics of several runs side by side, one row per eval step."""
    console.print(_compare_table(_read_runs(args.runs), args.metric))


def _add_schedule_args(parser: argparse.ArgumentParser, lr: float, warmup: int) -> None:
    """Batch size, learning rate, warmup and precision: the options both training runs take."""
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=TRAIN_DEFAULTS.dtype)


def _add_switch(parser: argparse.ArgumentParser, flag: str, help_: str) -> None:
    """An off-by-default ``--flag``."""
    parser.add_argument(flag, action="store_true", help=help_)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Seed, eval/log cadence, device, resuming and stopping: shared by both training runs."""
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS.seed)
    parser.add_argument(
        "--eval-every", type=int, default=TRAIN_DEFAULTS.eval_every, help="also how often to save"
    )
    parser.add_argument("--log-every", type=int, default=TRAIN_DEFAULTS.log_every)
    add_device_arg(parser)
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    parser.add_argument("--stop-after", type=int, default=None, help="stop early at this step")
    _add_set_arg(parser)


def _add_set_arg(parser: argparse.ArgumentParser) -> None:
    """``--set section.key=value``: change any experiment value, e.g. train.lr=1e-3."""
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override an experiment value (repeatable), e.g. train.lr=1e-3",
    )


def add_pretrain(commands: Subcommands) -> None:
    """Register ``eve pretrain``."""
    parser = commands.add_parser("pretrain", help="span-corruption pretraining")
    parser.set_defaults(run=pretrain)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="eve-32m")
    parser.add_argument("--data", default=defaults.BIN_DIR)
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE)
    parser.add_argument("--out", default=defaults.PRETRAIN_RUN)
    parser.add_argument("--tokens", default="850M", help="corpus tokens to read (sets --steps)")
    parser.add_argument("--steps", type=int, default=None, help="default: from --tokens")
    _add_schedule_args(parser, lr=TRAIN_DEFAULTS.lr, warmup=TRAIN_DEFAULTS.warmup_steps)
    parser.add_argument("--dropout", type=float, default=None, help="default: the preset's (0.1)")
    _add_switch(parser, "--rope", "rotary positions, no position tables")
    _add_switch(parser, "--qk-norm", "RMS-normalize queries and keys")
    _add_switch(parser, "--compile", "torch.compile: ~2x faster on MPS")
    _add_run_args(parser)


def add_finetune(commands: Subcommands) -> None:
    """Register ``eve finetune``."""
    parser = commands.add_parser("finetune", help="fine-tune questions and values")
    parser.set_defaults(run=finetune)
    start = parser.add_mutually_exclusive_group()  # one is required; finetune checks
    start.add_argument("--init", help="a pretraining checkpoint or an earlier finetune's model.pt")
    start.add_argument(
        "--scratch", choices=sorted(PRESETS), help="random init, to measure what pretraining buys"
    )
    parser.add_argument("--task", default=defaults.TASK_DIR, help="where train.jsonl is")
    parser.add_argument("--evals", default=defaults.EVALS_DIR, help="where val and held-out are")
    parser.add_argument("--tokenizer", default=defaults.TOKENIZER_FILE, help="for --scratch")
    parser.add_argument("--out", default=defaults.FINETUNE_RUN)
    parser.add_argument("--epochs", type=float, default=2.0, help="sets --steps")
    parser.add_argument("--steps", type=int, default=None, help="default: from --epochs")
    _add_schedule_args(parser, lr=FINETUNE_LR, warmup=FINETUNE_WARMUP)
    parser.add_argument("--dropout", type=float, default=None, help="default: the checkpoint's")
    _add_run_args(parser)


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
