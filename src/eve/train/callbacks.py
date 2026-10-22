"""The registered callbacks (kind "callback"): what a run does besides its optimizer steps.

The default run has all three, in this order: ``train_log`` (train losses and speed),
``evaluate`` (val metrics, the report's ``on_eval`` and the eval table) and ``checkpoint``
(``last.pt`` and ``model.pt``). They read their cadence from the run's ``TrainConfig``.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from rich.text import Text

from eve import progress
from eve.model.checkpoint import save_checkpoint
from eve.registry import register
from eve.train.loop import Report, Run, evaluate, rng_state, val_batches_of

logger = logging.getLogger(__name__)


@dataclass
class _Window:
    """Training stats since the last log line."""

    paused: float  # the run's paused seconds when the window started
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    tokens: int = 0
    steps: int = 0
    start: float = field(default_factory=time.time)

    def add(self, losses: dict[str, torch.Tensor], tokens: int) -> None:
        # Summed as tensors: calling .item() every step would make the CPU wait for the GPU.
        for name, value in losses.items():
            self.losses[name] = self.losses.get(name, 0.0) + value.detach().float()
        self.tokens += tokens
        self.steps += 1


@register("callback", "train_log")
class TrainLog:
    """Every ``log_every`` steps and at the run's last step: the mean losses, lr, gradient
    norm and speed since the last log line, logged (kind "train") and shown on the bar."""

    def on_start(self, run: Run, step: int) -> None:
        """Start the first window."""
        self.window = _Window(run.paused)

    def on_step(self, run: Run, step: int) -> None:
        """Add the step to the window; log and start a new one when due."""
        assert run.last is not None
        self.window.add(run.last.losses, run.last.tokens)
        if step % run.cfg.log_every == 0 or step == run.end:
            run.train_loss = self._report(run, step)
            self.window = _Window(run.paused)

    def on_end(self, run: Run, step: int) -> None:
        """Nothing to do: the last step was logged."""

    def _report(self, run: Run, step: int) -> float:
        """Log the window's mean losses and speed, and show them on the bar. Returns the mean
        total loss, the number that is trained on."""
        assert run.last is not None
        window = self.window
        elapsed = time.time() - window.start - (run.paused - window.paused)  # evals don't count
        means = {k: (v / window.steps).item() for k, v in window.losses.items()}
        loss = sum(means.values())
        tok_s = window.tokens / elapsed
        run.log.write(
            {
                "kind": "train",
                "step": step,
                "lr": run.last.lr,
                "grad_norm": run.last.grad_norm.item(),
                "tok_s": tok_s,
                "s_per_step": elapsed / window.steps,
                **means,
            }
        )
        lr = run.last.lr
        stats = [f"loss {loss:.4f}", f"lr {lr:.1e}", f"{tok_s / 1e3:.1f}k tok/s"]
        if run.bar.disable:  # not a terminal: plain lines for the log file
            eta = (run.end - step) * elapsed / window.steps
            logger.info(
                f"step {step:>6}/{run.end} | "
                + " | ".join(stats)
                + f" | eta {progress.format_duration(eta)}"
            )
        run.bar.set_stats(*stats)
        return loss


@register("callback", "evaluate")
class Evaluate:
    """At step 0 of a new run, every ``eval_every`` steps and at the last step of the schedule:
    the val metrics (``eval_fn`` on the first ``eval_batches`` val batches, kind "val") and what
    the report's ``on_eval`` returns (kind "samples"), as a row of the eval table. At the end:
    the report's summary and the last samples."""

    def on_start(self, run: Run, step: int) -> None:
        """Build the val batches, print the table's header, and evaluate a new run at step 0
        (the loss should be about ln(vocab_size))."""
        self.val_batches = val_batches_of(run.data, run.cfg)
        self.extras: dict[str, Any] = {}  # what on_eval returned at the last eval
        self.before: dict[str, Any] = {}  # the results of the eval before the last one
        run.bar.write(self._table(run.report).header())
        if step == 0:
            self._eval(run, step)

    def on_step(self, run: Run, step: int) -> None:
        """Evaluate when due."""
        if step % run.cfg.eval_every == 0 or step == run.cfg.steps:
            self._eval(run, step)

    def on_end(self, run: Run, step: int) -> None:
        """Show the summary and samples of the last eval."""
        if items := _summary(run.report, self._results(run)):
            progress.console.print()
            progress.console.print(progress.rows(items, indent=2))  # under the table
            progress.console.print()

    @staticmethod
    def _table(report: Report) -> progress.Table:
        """The eval table: step, time, train loss, then the report's columns."""
        return progress.Table(["step", "time", "train loss", *(c.label for c in report.columns)])

    def _results(self, run: Run) -> dict[str, Any]:
        """The last eval's val metrics and ``on_eval`` results together."""
        return {**run.last_val, **self.extras}

    def _eval(self, run: Run, step: int) -> None:
        """Evaluate, log the val metrics and what ``on_eval`` returns, and print the row."""
        self.before = self._results(run)
        run.last_val = evaluate(run.runner, run.eval_fn, self.val_batches, run.device, run.autocast)
        run.log.write({"kind": "val", "step": step, **run.last_val})
        if run.report.on_eval is not None:
            self.extras = run.report.on_eval(run.model)  # eager: generation changes shape
            run.log.write({"kind": "samples", "step": step, **self.extras})
        results = self._results(run)
        train_loss = "–" if run.train_loss is None else f"{run.train_loss:.4f}"
        cells = [
            Text(f"{step:,}", style="bold"),
            Text(progress.format_duration(time.time() - run.start), style="dim"),
            Text(train_loss, style="dim" if run.train_loss is None else ""),
        ]
        columns = [c.cell(results, self.before) for c in run.report.columns]
        run.bar.write(self._table(run.report).row(cells + columns))


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


@register("callback", "checkpoint")
class Checkpoint:
    """``last.pt`` every ``save_every`` steps and at the run's last step: everything a resumed
    run needs to go on exactly. At the end: ``model.pt`` if the run reached its last step,
    else how to continue."""

    def on_start(self, run: Run, step: int) -> None:
        """Nothing to do: there is nothing new to save yet."""

    def on_step(self, run: Run, step: int) -> None:
        """Save ``last.pt`` when due."""
        if step % run.cfg.save_every == 0 or step == run.end:
            state = {"step": step, "optimizer": run.optimizer.state_dict(), "rng": rng_state()}
            info = {
                **(run.meta or {}),
                "step": step,
                "train_config": run.cfg.to_dict(),
                **run.last_val,
            }
            save_checkpoint(run.out_dir / "last.pt", run.model, run.tokenizer, info, state)

    def on_end(self, run: Run, step: int) -> None:
        """Save ``model.pt`` if the run reached its last step, else say how to continue."""
        took = progress.format_duration(time.time() - run.start)
        if step == run.cfg.steps:
            info = {**(run.meta or {}), "step": step, **run.last_val}
            save_checkpoint(run.out_dir / "model.pt", run.model, run.tokenizer, info)
            progress.console.print()
            progress.success(f"done in {took}", run.out_dir / "model.pt")
        else:
            progress.console.print()
            progress.warn(f"stopped at step {step:,} after {took} · continue with --resume")
