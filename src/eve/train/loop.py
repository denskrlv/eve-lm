import json
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from rich.text import Text
from torch import nn

from eve import progress
from eve.components import DataModule, OptimizerFactory, Schedule
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve
from eve.runtime import eval_mode, to_device
from eve.tokenizer.bpe import BPETokenizer
from eve.train.config import DTYPES, TrainConfig
from eve.train.optim import adamw, lr_at

logger = logging.getLogger(__name__)

LossFn = Callable[[Eve, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
Autocast = Callable[[], AbstractContextManager[Any]]
RUN_FIELDS = (
    "eval_every",
    "eval_batches",
    "save_every",
    "log_every",
    "compile",
)  # may change on resume


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
    on_eval: Callable[[Eve], dict[str, Any]] | None = None
    summary: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None
    highlights: Sequence[tuple[str, str]] = ()


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


def _rng_state() -> dict[str, Any]:
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
    model: Eve,
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


@dataclass
class _Window:
    """Training stats since the last log line."""

    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    tokens: int = 0
    steps: int = 0
    start: float = field(default_factory=time.time)

    def add(self, losses: dict[str, torch.Tensor]) -> None:
        # Summed as tensors: calling .item() every step would make the CPU wait for the GPU.
        for name, value in losses.items():
            self.losses[name] = self.losses.get(name, 0.0) + value.detach().float()
        self.steps += 1


class _Evaluator:
    """Runs the evals of one training run, logs them and keeps the latest results."""

    def __init__(
        self,
        runner: Eve,
        model: Eve,
        eval_fn: LossFn,
        val_batches: Sequence[dict[str, torch.Tensor]],
        device: torch.device,
        autocast: Autocast,
        on_eval: Callable[[Eve], dict[str, Any]] | None,
    ) -> None:
        self.runner, self.model, self.eval_fn = runner, model, eval_fn
        self.val_batches, self.device, self.autocast = val_batches, device, autocast
        self.on_eval = on_eval
        self.last_val: dict[str, float] = {}
        self.extras: dict[str, Any] = {}  # what on_eval returned at the last eval
        self.before: dict[str, Any] = {}  # the results of the eval before the last one

    def run(self, step: int, log: MetricsLog) -> None:
        """Evaluate, then log the val metrics and what ``on_eval`` returns."""
        self.before = self.results
        self.last_val = evaluate(
            self.runner, self.eval_fn, self.val_batches, self.device, self.autocast
        )
        log.write({"kind": "val", "step": step, **self.last_val})
        if self.on_eval is not None:
            self.extras = self.on_eval(self.model)  # the eager model: generation changes shape
            log.write({"kind": "samples", "step": step, **self.extras})

    @property
    def results(self) -> dict[str, Any]:
        """The last eval's val metrics and ``on_eval`` results together."""
        return {**self.last_val, **self.extras}


def _autocast(device: torch.device, dtype_name: str) -> Autocast:
    dtype = DTYPES[dtype_name]
    if dtype is None:
        return nullcontext
    return lambda: torch.autocast(device.type, dtype=dtype)


def _val_batches(data: DataModule, cfg: TrainConfig) -> list[dict[str, torch.Tensor]]:
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
    runner: Eve,
    model: Eve,
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


def _report_train(
    window: _Window,
    step: int,
    end: int,
    lr: float,
    grad_norm: torch.Tensor,
    log: MetricsLog,
    bar: progress.Bar,
) -> float:
    """Log the mean losses and speed since the last report, and show them on the bar.
    Returns the mean total loss, the number that is trained on."""
    elapsed = time.time() - window.start
    means = {k: (v / window.steps).item() for k, v in window.losses.items()}
    loss = sum(means.values())
    tok_s = window.tokens / elapsed
    log.write(
        {
            "kind": "train",
            "step": step,
            "lr": lr,
            "grad_norm": grad_norm.item(),
            "tok_s": tok_s,
            "s_per_step": elapsed / window.steps,
            **means,
        }
    )
    stats = [f"loss {loss:.4f}", f"lr {lr:.1e}", f"{tok_s / 1e3:.1f}k tok/s"]
    if bar.disable:  # not a terminal: plain lines for the log file
        eta = (end - step) * elapsed / window.steps
        logger.info(
            f"step {step:>6}/{end} | "
            + " | ".join(stats)
            + f" | eta {progress.format_duration(eta)}"
        )
    bar.set_stats(*stats)
    return loss


def _start_step(
    model: Eve,
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


def _prepare(
    model: Eve, cfg: TrainConfig, device: torch.device, optimizer: OptimizerFactory
) -> tuple[Eve, torch.optim.Optimizer, Autocast]:
    """The model to run forward passes with, the optimizer, and the autocast context."""
    # torch.compile traces the forward pass once and fuses its many small operations (each
    # LayerNorm, GELU and RMSNorm here is 5-10 of them) into a few kernels: about 2x faster on
    # MPS, same losses. The first step takes ~20 s to compile. ``model`` itself stays eager for
    # ``on_eval``, whose generation changes shape every token, and for saving.
    runner = cast(Eve, torch.compile(model)) if cfg.compile else model
    return runner, optimizer(model, cfg, device), _autocast(device, cfg.dtype)


@dataclass
class _Run:
    """What one run's periodic logs, evals and saves need."""

    cfg: TrainConfig
    end: int  # the step this run stops at
    out_dir: Path
    model: Eve
    tokenizer: BPETokenizer
    optimizer: torch.optim.Optimizer
    meta: dict[str, Any] | None
    log: MetricsLog
    bar: progress.Bar
    evaluator: _Evaluator
    report: Report
    schedule: Schedule
    train_loss: float | None = None  # the mean of the last logged window
    start: float = field(default_factory=time.time)

    @property
    def table(self) -> progress.Table:
        """The eval table: step, time, train loss, then the report's columns."""
        labels = ["step", "time", "train loss", *(c.label for c in self.report.columns)]
        return progress.Table(labels)


def _eval(run: _Run, step: int) -> None:
    """Evaluate and print the eval's row of the table."""
    run.evaluator.run(step, run.log)
    results, before = run.evaluator.results, run.evaluator.before
    train_loss = "–" if run.train_loss is None else f"{run.train_loss:.4f}"
    cells = [
        Text(f"{step:,}", style="bold"),
        Text(progress.format_duration(time.time() - run.start), style="dim"),
        Text(train_loss, style="dim" if run.train_loss is None else ""),
    ]
    run.bar.write(run.table.row(cells + [c.cell(results, before) for c in run.report.columns]))


def _after_step(
    run: _Run, step: int, lr: float, grad_norm: torch.Tensor, window: _Window
) -> _Window:
    """Log, evaluate and save when ``step`` is due; returns the stats window to go on with."""
    if step % run.cfg.log_every == 0 or step == run.end:
        run.train_loss = _report_train(window, step, run.end, lr, grad_norm, run.log, run.bar)
        window = _Window()
    paused = time.time()
    if step % run.cfg.eval_every == 0 or step == run.cfg.steps:
        _eval(run, step)
    if step % run.cfg.save_every == 0 or step == run.end:
        _save_last(run, step)
    window.start += time.time() - paused  # eval and saving don't count towards tok/s
    return window


def train(
    model: Eve,
    loss_fn: LossFn,
    data: DataModule,
    cfg: TrainConfig,
    out_dir: str | Path,
    tokenizer: BPETokenizer,
    device: str | torch.device,
    report: Report | None = None,
    eval_fn: LossFn | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    meta: dict[str, Any] | None = None,
    optimizer: OptimizerFactory = adamw,
    schedule: Schedule = lr_at,
) -> dict[str, float]:
    """The training loop for pretraining and fine-tuning (book ch. 5.2 ``train_model_simple``
    plus App. D's warmup, cosine decay and gradient clipping).

    Each step: a batch -> ``loss_fn`` gives named losses -> their sum is backpropagated ->
    gradients are clipped to norm ``grad_clip`` -> ``optimizer`` step at ``schedule(step)``
    (default: AdamW, warmup then cosine decay; see ``eve.train.optim``).

    ``out_dir`` gets:
    - ``metrics.jsonl``: one JSON line per log (``kind`` "train"), eval ("val") and sample
      set ("samples", whatever ``report.on_eval`` returns);
    - ``last.pt``: model + optimizer + step + RNG states, written every ``save_every`` steps;
    - ``model.pt``: the final weights without the optimizer, for ``load_pretrained`` (box 7).

    Evals run ``eval_fn`` (default: ``loss_fn``) on the val batches. It may return extra
    metrics (e.g. an accuracy) that are logged with a ``val_`` prefix but never trained on.
    Each eval prints one row of a table (``report.columns``); samples and the ``report.summary``
    lines are shown once, when the run ends.

    ``meta`` (plain values) is added to both checkpoints' meta, e.g. what kind of run made them.
    A dataset with a ``set_epoch`` method (``TaskDataset``) is told the epoch of each batch, so
    it can shuffle differently per epoch.

    ``resume=True`` continues from ``last.pt`` exactly: the same batches, dropout masks and
    optimizer state, so the losses match an uninterrupted run. ``stop_after`` ends the run
    early at that step (saving ``last.pt``), as if it had been interrupted.

    Returns the last val losses.
    """
    out_dir, device = Path(out_dir), torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    runner, opt, autocast = _prepare(model, cfg, device, optimizer)
    step = _start_step(model, opt, out_dir, cfg, resume)
    end = cfg.steps if stop_after is None else min(stop_after, cfg.steps)
    report = report or Report()
    val_batches = _val_batches(data, cfg)
    evaluator = _Evaluator(
        runner, model, eval_fn or loss_fn, val_batches, device, autocast, report.on_eval
    )
    bar = progress.bar(end, out_dir.name, "steps", initial=step)  # to where this run stops
    log = MetricsLog(out_dir / "metrics.jsonl")
    run = _Run(
        cfg, end, out_dir, model, tokenizer, opt, meta, log, bar, evaluator, report, schedule
    )
    _start_log(run, step)

    step = _run_steps(run, runner, loss_fn, data, device, autocast, step)
    bar.close()
    _finish(run, step)
    return evaluator.last_val


def _start_log(run: _Run, step: int) -> None:
    """Print the eval table's header. A new run (``step`` 0) also starts an empty log with its
    configs and a first eval; a resumed run keeps the log ``_resume`` trimmed."""
    run.bar.write(run.table.header())
    if step != 0:
        return
    run.log.clear()
    run.log.write(
        {
            "kind": "start",
            "train_config": run.cfg.to_dict(),
            "model_config": run.model.cfg.to_dict(),
        }
    )
    _eval(run, step)  # at step 0 the loss should be about ln(vocab_size)


def _run_steps(
    run: _Run,
    runner: Eve,
    loss_fn: LossFn,
    data: DataModule,
    device: torch.device,
    autocast: Autocast,
    step: int,
) -> int:
    """Train from ``step`` to ``run.end``, logging, evaluating and saving on the way. Returns
    the step reached."""
    window = _Window()
    while step < run.end:
        batch = _train_batch(data, run.cfg, step)
        window.tokens += data.batch_tokens(batch)
        lr = run.schedule(step, run.cfg)
        losses, grad_norm = _train_step(
            runner, run.model, loss_fn, run.optimizer, batch, lr, run.cfg, device, autocast
        )
        step += 1
        window.add(losses)
        run.bar.update()
        window = _after_step(run, step, lr, grad_norm, window)
    return step


def _save_last(run: _Run, step: int) -> None:
    """``last.pt``: everything ``_resume`` needs to continue exactly where the run is now."""
    state = {"step": step, "optimizer": run.optimizer.state_dict(), "rng": _rng_state()}
    info = {
        **(run.meta or {}),
        "step": step,
        "train_config": run.cfg.to_dict(),
        **run.evaluator.last_val,
    }
    save_checkpoint(run.out_dir / "last.pt", run.model, run.tokenizer, info, state)


def _finish(run: _Run, step: int) -> None:
    """Save ``model.pt`` if the run reached its last step, else say how to continue. Then show
    the summary and samples of the last eval."""
    took = progress.format_duration(time.time() - run.start)
    if step == run.cfg.steps:
        info = {**(run.meta or {}), "step": step, **run.evaluator.last_val}
        save_checkpoint(run.out_dir / "model.pt", run.model, run.tokenizer, info)
        progress.console.print()
        progress.success(f"done in {took}", run.out_dir / "model.pt")
    else:
        progress.console.print()
        progress.warn(f"stopped at step {step:,} after {took} · continue with --resume")
    if items := _summary(run.report, run.evaluator.results):
        progress.console.print()
        progress.console.print(progress.rows(items, indent=2))  # under the table
        progress.console.print()


def _summary(report: Report, results: dict[str, Any]) -> list[tuple[str, str | Text]]:
    """The labelled lines shown at the end: the report's summary, then the samples with the
    report's highlights."""
    items: list[tuple[str, str | Text]] = list(report.summary(results) if report.summary else [])
    for i, line in enumerate(results.get("samples", [])):
        text = Text(line)
        for pattern, style in report.highlights:
            text.highlight_regex(pattern, style)
        items.append(("samples" if i == 0 else "", text))
    return items


def _resume(
    model: Eve,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    cfg: TrainConfig,
) -> int:
    """Restore the model, optimizer and RNG states from ``last.pt`` and return its step."""
    ckpt = load_checkpoint(out_dir / "last.pt")
    if ckpt.model.cfg != model.cfg:
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
