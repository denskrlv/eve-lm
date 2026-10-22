"""The trainer: one optimizer step after another, with the components it is given (see
``eve.components``). It knows no model, data or loss: what to log, evaluate and save at each
step is up to its callbacks (``eve.train.callbacks``)."""

import json
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from rich.text import Text
from torch import nn

from eve import progress, registry
from eve.components import Callback, DataModule, LossFn, OptimizerFactory, Schedule
from eve.model.checkpoint import load_checkpoint
from eve.runtime import eval_mode, to_device
from eve.train.config import DTYPES, TrainConfig

if TYPE_CHECKING:
    from eve.tokenizer.bpe import BPETokenizer

logger = logging.getLogger(__name__)

Autocast = Callable[[], AbstractContextManager[Any]]
RUN_FIELDS = (
    "eval_every",
    "eval_batches",
    "save_every",
    "log_every",
    "compile",
)  # may change on resume
# What a run is given when the caller passes none, by registered name.
DEFAULT_OPTIMIZER, DEFAULT_SCHEDULE = "adamw", "warmup_cosine"
DEFAULT_CALLBACKS = ("train_log", "evaluate", "checkpoint")


@dataclass(frozen=True)
class Column:
    """A column of the eval table: its label and where its value is in an eval's results.

    ``key`` is a val metric (``val_lm``) or ``group.name`` inside a dict that ``on_eval``
    returned (``heldout.macro_recall``).
    """

    label: str
    key: str

    def value(self, results: dict[str, Any]) -> float | None:
        """This column's value in ``results``, or None if they don't have it."""
        value: Any = results
        for part in self.key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return None if value is None else float(value)

    def cell(self, results: dict[str, Any], before: dict[str, Any] | None = None) -> Text:
        """The value formatted like the training log ("–" if missing): green if it got better
        since the eval ``before``, red if it got worse."""
        value = self.value(results)
        if value is None:
            return Text("–", style="dim")
        text = progress.format_metric(self.key.rsplit(".", 1)[-1], value)
        old = self.value(before or {})
        if old is None or old == value:
            return Text(text)
        better = (value > old) == progress.higher_is_better(self.key)
        return Text(text, style="green" if better else "red")


@dataclass(frozen=True)
class Report:
    """What a run shows besides its progress bar.

    - ``columns``: the eval table's columns after step, time and train loss; a row per eval.
    - ``on_eval``: runs on the eager model at each eval. What it returns is logged (kind
      "samples"), the columns can read it, and its ``samples`` lines are shown at the end.
    - ``summary``: labelled lines shown at the end, made from the last eval's results.
    - ``highlights``: (regex, rich style) pairs that color parts of the sample lines, e.g. the
      filled-in spans.
    """

    columns: Sequence[Column] = ()
    on_eval: Callable[[Any], dict[str, Any]] | None = None
    summary: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None
    highlights: Sequence[tuple[str, str]] = ()


class MetricsLog:
    """``metrics.jsonl``: one JSON record per line, appended as the run goes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def clear(self) -> None:
        """Start an empty log (a new run)."""
        self.path.write_text("")

    def write(self, record: dict[str, Any]) -> None:
        """Append one record."""
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def model_config(model: nn.Module) -> dict[str, Any]:
    """The model's config as plain values (``model.cfg.to_dict()``), for logs and checks."""
    config: dict[str, Any] = getattr(model, "cfg").to_dict()  # noqa: B009 (hidden from mypy)
    return config


@dataclass
class StepStats:
    """What the last optimizer step did: its losses, learning rate, gradient norm before
    clipping, and how many real tokens its batch had."""

    losses: dict[str, torch.Tensor]
    lr: float
    grad_norm: torch.Tensor
    tokens: int


@dataclass
class Run:
    """One training run as its callbacks see it: its config and components, and what the
    callbacks share (the last step's stats, the last eval's val metrics, the train loss)."""

    cfg: TrainConfig
    end: int  # the step this run stops at
    out_dir: Path
    model: nn.Module  # eager: for on_eval, whose generation changes shape, and for saving
    runner: nn.Module  # what the loss runs on: ``model``, or its torch.compile'd wrapper
    tokenizer: "BPETokenizer"  # saved with every checkpoint
    optimizer: torch.optim.Optimizer
    data: DataModule
    eval_fn: LossFn
    report: Report
    device: torch.device
    autocast: Autocast
    meta: dict[str, Any] | None
    log: MetricsLog
    bar: progress.Bar
    last: StepStats | None = None
    last_val: dict[str, float] = field(default_factory=dict)  # set by the evaluate callback
    train_loss: float | None = None  # the mean of the last logged window (train_log)
    paused: float = 0.0  # seconds spent in callbacks, which tok/s leaves out
    start: float = field(default_factory=time.time)


def batch_indices(step: int, n_items: int, cfg: TrainConfig) -> np.ndarray:
    """The dataset indices of batch ``step``: each epoch is a new shuffle seeded by
    ``(seed, epoch)``, cut into whole batches (the last partial one is dropped). The order
    depends only on the step, so a resumed run sees exactly the batches it would have seen."""
    per_epoch = n_items // cfg.batch_size
    if per_epoch == 0:
        raise ValueError(f"{n_items} items is less than one batch of {cfg.batch_size}")
    epoch, i = divmod(step, per_epoch)
    order = np.random.default_rng((cfg.seed, epoch)).permutation(n_items)
    return order[i * cfg.batch_size : (i + 1) * cfg.batch_size]


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: LossFn,
    batches: Sequence[dict[str, Any]],
    device: torch.device,
    autocast: Autocast,
) -> dict[str, float]:
    """Mean of each loss over the val batches, with dropout off (book ch. 5.1.3)."""
    totals: dict[str, torch.Tensor] = {}
    with eval_mode(model):
        for batch in batches:  # kept alive by the caller, so the copy may run in the background
            with autocast():
                losses = loss_fn(model, to_device(batch, device, non_blocking=True))
            for name, loss in losses.items():
                totals[name] = totals.get(name, 0.0) + loss.float()
    return {f"val_{name}": (total / len(batches)).item() for name, total in totals.items()}


def _autocast(device: torch.device, dtype_name: str) -> Autocast:
    dtype = DTYPES[dtype_name]
    if dtype is None:
        return nullcontext
    return lambda: torch.autocast(device.type, dtype=dtype)


def val_batches_of(data: DataModule, cfg: TrainConfig) -> list[dict[str, torch.Tensor]]:
    """The first ``eval_batches`` whole batches of val: the same ones at every eval."""
    size, val = cfg.batch_size, data.val
    batches = [
        data.collate([val[j] for j in range(i * size, (i + 1) * size)])
        for i in range(min(cfg.eval_batches, len(val) // size))
    ]
    if not batches:
        raise ValueError(f"val data has {len(val)} items, less than one batch")
    return batches


def _train_batch(data: DataModule, cfg: TrainConfig, step: int) -> dict[str, torch.Tensor]:
    """Batch ``step``, after telling a train set with ``set_epoch`` which epoch it is in."""
    train_data = data.train
    idx = batch_indices(step, len(train_data), cfg)
    if hasattr(train_data, "set_epoch"):
        train_data.set_epoch(step // (len(train_data) // cfg.batch_size))
    return data.collate([train_data[int(j)] for j in idx])


def _train_step(
    runner: nn.Module,
    model: nn.Module,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    lr: float,
    cfg: TrainConfig,
    device: torch.device,
    autocast: Autocast,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One optimizer step at learning rate ``lr``. Returns the losses and the gradient norm
    before clipping."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    with autocast():
        losses = loss_fn(runner, to_device(batch, device, non_blocking=True))  # train keeps batch
    loss = cast(torch.Tensor, sum(losses.values()))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return losses, grad_norm


def _prepare(
    model: nn.Module, cfg: TrainConfig, device: torch.device, optimizer: OptimizerFactory
) -> tuple[nn.Module, torch.optim.Optimizer, Autocast]:
    """The model to run forward passes with, the optimizer, and the autocast context."""
    # torch.compile traces the forward pass once and fuses its many small operations (each
    # LayerNorm, GELU and RMSNorm here is 5-10 of them) into a few kernels: about 2x faster on
    # MPS, same losses. The first step takes ~20 s to compile. ``model`` itself stays eager for
    # ``on_eval``, whose generation changes shape every token, and for saving.
    runner = cast(nn.Module, torch.compile(model)) if cfg.compile else model
    return runner, optimizer(model, cfg, device), _autocast(device, cfg.dtype)


def _start_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    cfg: TrainConfig,
    resume: bool,
) -> int:
    """The step to start at: 0, or where ``last.pt`` left off when resuming."""
    if resume and (out_dir / "last.pt").exists():
        return _resume(model, optimizer, out_dir, cfg)
    if resume:
        logger.info(f"nothing to resume in {out_dir}: starting a new run")
    return 0


def _timed(run: Run, hook: Callable[[Run, int], None], step: int) -> None:
    """Call a callback hook; its time counts as paused, not as training."""
    start = time.time()
    hook(run, step)
    run.paused += time.time() - start


@registry.register("trainer", "default")
def train(
    model: nn.Module,
    loss_fn: LossFn,
    data: DataModule,
    cfg: TrainConfig,
    out_dir: str | Path,
    tokenizer: "BPETokenizer",
    device: str | torch.device,
    report: Report | None = None,
    eval_fn: LossFn | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    meta: dict[str, Any] | None = None,
    optimizer: OptimizerFactory | None = None,
    schedule: Schedule | None = None,
    callbacks: Sequence[Callback] | None = None,
    experiment: dict[str, Any] | None = None,
) -> dict[str, float]:
    """The training loop for pretraining and fine-tuning (book ch. 5.2 ``train_model_simple``
    plus App. D's warmup, cosine decay and gradient clipping).

    Each step: a batch of ``data`` -> ``loss_fn`` gives named losses -> their sum is
    backpropagated -> gradients are clipped to norm ``grad_clip`` -> ``optimizer`` step at
    ``schedule(step)``. Without them: AdamW, warmup then cosine decay (the ``DEFAULT_*`` names).

    ``out_dir`` gets ``metrics.jsonl``: a "start" record with the configs (``experiment``, the
    resolved run config, if given), then one JSON line per log, eval and sample set. The
    callbacks (default: ``train_log``, ``evaluate``, ``checkpoint``) write those and the
    checkpoints: ``last.pt`` (model + optimizer + step + RNG states) every ``save_every``
    steps, and ``model.pt``, the final weights without the optimizer, for ``load_pretrained``
    (box 7).

    Evals run ``eval_fn`` (default: ``loss_fn``) on the first val batches. It may return extra
    metrics (e.g. an accuracy) that are logged with a ``val_`` prefix but never trained on.
    ``report`` adds the eval table's columns, samples and summary.

    ``meta`` (plain values) is added to both checkpoints' meta, e.g. what kind of run made them.
    A train set with a ``set_epoch`` method (``TaskDataset``) is told the epoch of each batch.

    ``resume=True`` continues from ``last.pt`` exactly: the same batches, dropout masks and
    optimizer state, so the losses match an uninterrupted run. ``stop_after`` ends the run
    early at that step (saving ``last.pt``), as if it had been interrupted.

    Returns the last val metrics.
    """
    optimizer = optimizer or registry.build("optimizer", DEFAULT_OPTIMIZER)
    schedule = schedule or registry.build("schedule", DEFAULT_SCHEDULE)
    if callbacks is None:
        callbacks = [registry.build("callback", name)() for name in DEFAULT_CALLBACKS]
    out_dir, device = Path(out_dir), torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    runner, opt, autocast = _prepare(model, cfg, device, optimizer)
    step = _start_step(model, opt, out_dir, cfg, resume)
    end = cfg.steps if stop_after is None else min(stop_after, cfg.steps)
    bar = progress.bar(end, out_dir.name, "steps", initial=step)  # to where this run stops
    log = MetricsLog(out_dir / "metrics.jsonl")
    run = Run(
        cfg, end, out_dir, model, runner, tokenizer, opt, data, eval_fn or loss_fn,
        report or Report(), device, autocast, meta, log, bar,
    )  # fmt: skip
    if step == 0:  # a resumed run keeps the log ``_resume`` trimmed
        log.clear()
        start = {"train_config": cfg.to_dict(), "model_config": model_config(model)}
        log.write({"kind": "start", **start, **({"experiment": experiment} if experiment else {})})
    for callback in callbacks:
        _timed(run, callback.on_start, step)

    while step < end:
        batch = _train_batch(data, cfg, step)
        lr = schedule(step, cfg)
        losses, grad_norm = _train_step(
            runner, model, loss_fn, opt, batch, lr, cfg, device, autocast
        )
        step += 1
        run.last = StepStats(losses, lr, grad_norm, data.batch_tokens(batch))
        bar.update()
        for callback in callbacks:
            _timed(run, callback.on_step, step)
    bar.close()
    for callback in reversed(callbacks):
        callback.on_end(run, step)
    return run.last_val


def _resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    cfg: TrainConfig,
) -> int:
    """Restore the model, optimizer and RNG states from ``last.pt`` and return its step."""
    ckpt = load_checkpoint(out_dir / "last.pt")
    if ckpt.model.cfg.to_dict() != model_config(model):
        raise ValueError(f"{out_dir}/last.pt has a different model config")
    saved, given = ckpt.meta["train_config"], cfg.to_dict()
    if changed := [k for k in given if k not in RUN_FIELDS and saved.get(k) != given[k]]:
        raise ValueError(
            f"can't resume with a different {', '.join(changed)}: "
            + ", ".join(f"{k} was {saved.get(k)!r}" for k in changed)
        )
    if ckpt.train_state is None:
        raise ValueError(f"{out_dir}/last.pt has no optimizer and RNG state to resume from")
    model.load_state_dict(ckpt.model.state_dict())
    optimizer.load_state_dict(ckpt.train_state["optimizer"])  # moves its state to the device
    _set_rng_state(ckpt.train_state["rng"])
    step: int = ckpt.train_state["step"]

    # Drop metrics logged after the checkpoint: those steps run again now.
    path = out_dir / "metrics.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    kept = [ln for ln in lines if json.loads(ln).get("step", 0) <= step]
    path.write_text("".join(ln + "\n" for ln in kept))
    logger.info(f"resumed {out_dir} at step {step:,}")
    return step
