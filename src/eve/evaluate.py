from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eve.config import INTENTS
from eve.data.render import TaskCollator, TaskDataset, TaskExample
from eve.data.schema import Tool
from eve.model.checkpoint import Checkpoint, load_checkpoint
from eve.train.finetune import (
    TASK,
    VAL_SEED,
    Outcome,
    Tally,
    in_batches,
    load_examples,
    outcomes,
    tally,
)

SETS = ("val", "heldout-templates", "massive-test")  # the task files scored by default
SEEDS = {"val": VAL_SEED}  # tool-shuffle seeds that differ from TaskDataset's default
MODEL_FILE = "model.pt"  # what a run directory holds the final model in


@dataclass(frozen=True)
class ToolResult:
    """How a model did on the examples whose gold calls use one tool: CALLs, and ASKs that
    name the tool but miss a value."""

    tool: str
    examples: int
    intent_acc: float  # the gold intent answered
    tools_acc: float  # the right tools in the right order, then STOP
    exact_acc: float  # both
    answered: dict[str, int]  # intent answered -> examples

    @classmethod
    def from_tally(cls, tool: str, counts: Tally) -> "ToolResult":
        """The result from the ``Tally`` of the tool's examples."""
        scores, total = counts.scores(), sum(counts.seen[intent] for intent in INTENTS)
        return cls(
            tool=tool,
            examples=total,
            intent_acc=sum(counts.right[intent] for intent in INTENTS) / total,
            tools_acc=scores["tools_acc"],
            exact_acc=scores["exact_acc"],
            answered={s: sum(counts.answered[g, s] for g in INTENTS) for s in INTENTS},
        )


@dataclass(frozen=True)
class SetResult:
    """How a model did on one task file."""

    examples: int
    scores: dict[str, float]  # as fine-tuning's evals log them, see finetune.score
    confusion: dict[str, dict[str, int]]  # gold intent -> answered intent -> examples
    by_tool: list[ToolResult]  # worst exact accuracy first

    def to_json(self, by_tool: bool = False) -> dict[str, Any]:
        """The result as plain values; the per-tool results only with ``by_tool``."""
        out = {"examples": self.examples, "scores": self.scores, "confusion": self.confusion}
        return out | ({"by_tool": [vars(r) for r in self.by_tool]} if by_tool else {})


def model_file(path: str | Path) -> Path:
    """A checkpoint path as given, or the final model of a run directory."""
    path = Path(path)
    return path / MODEL_FILE if path.is_dir() else path


def load_questions_model(path: str | Path, device: str | torch.device = "cpu") -> Checkpoint:
    """A fine-tuned checkpoint; a pretraining one is refused, its question heads are random."""
    ckpt = load_checkpoint(model_file(path), device)
    if ckpt.meta.get("task") != TASK:
        raise ValueError(f"{path} is a pretraining checkpoint: its question heads are untrained")
    return ckpt


def evaluate_set(
    ckpt: Checkpoint, path: Path, library: dict[str, Tool], batch_size: int
) -> SetResult:
    """Score the heads on every example of one task file, in order.

    Each file keeps the tool shuffle fine-tuning's evals used (``SEEDS``), so a model's val
    and held-out scores here are the ones its run logged at its last eval.
    """
    cfg = ckpt.model.cfg
    examples = load_examples(path, library)
    data = TaskDataset(examples, ckpt.tokenizer, cfg, seed=SEEDS.get(path.stem, 0))
    answers = outcomes(ckpt.model, in_batches(data, TaskCollator(cfg), batch_size))
    counts = tally(answers)
    confusion = {gold: {said: counts.answered[gold, said] for said in INTENTS} for gold in INTENTS}
    return SetResult(len(examples), counts.scores(), confusion, by_tool(examples, answers))


def by_tool(examples: Sequence[TaskExample], answers: Sequence[Outcome]) -> list[ToolResult]:
    """A result per tool in the gold calls, worst exact accuracy first. An example with two
    calls counts for both tools; one without calls (an unclear ASK, UNSUPPORTED, NONE) for none."""
    tallies: defaultdict[str, Tally] = defaultdict(Tally)
    for example, outcome in zip(examples, answers, strict=True):
        for tool in dict.fromkeys(call.tool for call in example.calls):
            tallies[tool].add(outcome)
    results = [ToolResult.from_tally(tool, counts) for tool, counts in tallies.items()]
    return sorted(results, key=lambda result: (result.exact_acc, result.tool))


def task_files(task_dir: str | Path, sets: Sequence[str] = SETS) -> dict[str, Path]:
    """The .jsonl file of each set name in ``task_dir``; all of them must exist."""
    paths = {name: Path(task_dir) / f"{name}.jsonl" for name in sets}
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"no task file {', '.join(missing)}")
    return paths
