"""M9: make the probabilities mean what they say.

A model trained with cross-entropy is usually too sure of itself: of the answers it gives 95%,
far fewer than 95% are right. Temperature scaling (Guo et al., 2017, "On Calibration of Modern
Neural Networks") fixes this with one number per answer: divide the logits by a temperature T
before the softmax, with T chosen to minimize the negative log-likelihood of a held-out set.
T > 1 spreads the probabilities out; the most likely answer, and so every accuracy, stays the
same.

Eve has three kinds of answers, each with its own temperature (``Calibration``): the intent,
each call slot ("what is call #i?") and each token of an argument value. A value's
probability is the product of its tokens'.

And two regimes (``Calibrations``): requests whose tools all appear in the training data
(familiar), and requests with a tool the model never saw (new). On new tools the model is as
sure as on familiar ones but right far less often, so each regime gets its own temperatures,
fitted on its own examples: familiar ones of val, new ones of the probe split.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eve.data.jsonl import read_jsonl
from eve.data.modules import in_batches, load_examples
from eve.data.render import IGNORE, TaskCollator, TaskDataset
from eve.data.schema import Tool
from eve.evaluate import SEEDS
from eve.infer.predict import Calibration, Calibrations
from eve.model.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from eve.model.eve import Eve, Output
from eve.runtime import eval_mode, to_device
from eve.tasks.questions import forward_batch

HEADS = ("intent", "tools", "values")  # the answers that get a temperature, as in Calibration
REGIMES = ("familiar", "new")  # as in Calibrations
GRID = tuple(0.25 * 32 ** (i / 80) for i in range(81))  # temperatures tried: 0.25 to 8, ~4% apart
BINS = 15  # confidence bins of the expected calibration error, as in Guo et al.


def known_tools(train_path: str | Path, library: dict[str, Tool]) -> frozenset[str]:
    """The fingerprint of every tool schema in a training file: a library name or a whole
    schema (external datasets). What ``Calibrations.is_familiar`` checks a request against."""
    names, schemas = set(), {}
    for row in read_jsonl(train_path):
        for tool in row["tools"]:
            if isinstance(tool, str):
                names.add(tool)
            else:
                schemas[repr(sorted(tool.items()))] = tool
    tools = [library[name] for name in names] + [Tool.from_dict(t) for t in schemas.values()]
    return frozenset(tool.fingerprint() for tool in tools)


@dataclass(frozen=True)
class TaskSet:
    """One task file in batches, with which of its examples have only familiar tools."""

    batches: list[dict[str, torch.Tensor]]
    familiar: list[torch.Tensor]  # per batch, (B,) bool

    def counts(self) -> dict[str, int]:
        """How many examples each regime has."""
        familiar = sum(int(f.sum()) for f in self.familiar)
        return {"familiar": familiar, "new": sum(f.numel() for f in self.familiar) - familiar}

    def only(self, regime: str) -> list[torch.Tensor]:
        """Per batch, which examples belong to ``regime``."""
        return [f if regime == "familiar" else ~f for f in self.familiar]


def load_set(
    ckpt: Checkpoint,
    path: Path,
    library: dict[str, Tool],
    batch_size: int,
    calibrations: Calibrations,
) -> TaskSet:
    """One task file's examples in batches, with the tool shuffle ``eve evaluate`` uses."""
    cfg = ckpt.model.cfg
    examples = load_examples(path, library)
    data = TaskDataset(examples, ckpt.tokenizer, cfg, seed=SEEDS.get(path.stem, 0))
    familiar = torch.tensor([calibrations.is_familiar(e.tools) for e in examples])
    return TaskSet(
        in_batches(data, TaskCollator(cfg), batch_size), list(familiar.split(batch_size))
    )


@dataclass(frozen=True)
class _Answered:
    """One head's answers in a batch: a row of logits per answer (an intent, a slot, a value
    token), the gold labels, which example each row is from, and which answer it belongs to
    (``group``): a value is several token rows, the other heads one row per answer."""

    logits: torch.Tensor  # (n, classes)
    labels: torch.Tensor  # (n,)
    example: torch.Tensor  # (n,) the row's example in the batch
    group: torch.Tensor  # (n,) 0, 1, … in order; rows of one answer are next to each other

    def rows(self, keep: torch.Tensor) -> "_Answered":
        """Only the rows of the examples ``keep`` (B,) marks, regrouped from 0."""
        mask = keep[self.example]
        group = self.group[mask]
        if group.numel():
            group = torch.unique_consecutive(group, return_inverse=True)[1]
        return _Answered(self.logits[mask], self.labels[mask], self.example[mask], group)


def _answered(out: Output, batch: dict[str, torch.Tensor], stop: int) -> dict[str, _Answered]:
    """The answers of each head in one batch that calibration counts.

    - intent: every example.
    - tools: the slots an app reads: each gold call's slot and the STOP after the last one (an
      example with 2 calls counts 3 slots). The STOPs after that are near certain and would
      only make the model look well calibrated.
    - values: every token of every argument value of the gold calls (``value_hits``'s values).
    """
    assert out.questions is not None and out.lm is not None
    intent = out.questions.intent.float()
    device = intent.device
    examples = torch.arange(intent.shape[0], device=device)

    slots = batch["slots"]
    labelled = slots != IGNORE
    calls = (slots != stop).long().cumprod(-1)  # 1 while every slot so far is a call
    read = labelled & F.pad(calls[:, :-1], (1, 0), value=1).bool()  # up to the first STOP
    slot_example = examples[:, None].expand_as(slots)[read]

    labels = batch["dec_labels"]
    tokens = labels != IGNORE
    starts = tokens & ~F.pad(tokens[:, :-1], (1, 0), value=False)
    value = starts.flatten().long().cumsum(0).view_as(labels) - 1  # which value, over the batch
    return {
        "intent": _Answered(intent, batch["intent"], examples, examples),
        "tools": _Answered(
            out.questions.slots.float()[read],
            slots[read],
            slot_example,
            torch.arange(slot_example.numel(), device=device),
        ),
        "values": _Answered(
            out.lm.float()[tokens],
            labels[tokens],
            examples[:, None].expand_as(labels)[tokens],
            value[tokens],
        ),
    }


def _answers(
    model: Eve, batches: Sequence[dict[str, torch.Tensor]]
) -> Iterator[dict[str, _Answered]]:
    """Every batch's answers, with dropout off."""
    with torch.no_grad(), eval_mode(model) as device:
        for batch in batches:
            batch = to_device(batch, device)
            yield _answered(forward_batch(model, batch), batch, model.cfg.max_tools)


def fit(
    model: Eve,
    batches: Sequence[dict[str, torch.Tensor]],
    keep: Sequence[torch.Tensor] | None = None,
) -> Calibration:
    """The temperature of each head that minimizes its negative log-likelihood on ``batches``
    (only the examples ``keep`` marks per batch, if given), from ``GRID``. A head without
    answers (no calls at all) keeps 1.

    For values this is the NLL of each gold token, as in training (teacher forcing), without
    the masks that inference adds (``predict._allowed``): they only remove tokens the trained
    decoder gives almost no probability to.
    """
    temperatures = torch.tensor(GRID)
    nll = {head: torch.zeros(len(GRID), dtype=torch.float64) for head in HEADS}
    for i, answered in enumerate(_answers(model, batches)):
        for head, a in answered.items():
            if keep is not None:
                a = a.rows(keep[i].to(a.example.device))
            if not a.labels.numel():
                continue
            losses = [F.cross_entropy(a.logits / t, a.labels, reduction="sum") for t in GRID]
            nll[head] += torch.stack(losses).cpu().double()  # one sync per head and batch
    best = {
        head: float(temperatures[int(losses.argmin())]) if losses.any() else 1.0
        for head, losses in nll.items()
    }
    return Calibration(**best)


@dataclass(frozen=True)
class Reliability:
    """How sure one head was of each of its answers (the most likely one), and whether it was
    right: what a reliability diagram plots."""

    confidence: torch.Tensor  # (n,) probability of the answer given, in [0, 1]
    right: torch.Tensor  # (n,) bool

    @property
    def n(self) -> int:
        """How many answers."""
        return int(self.confidence.numel())

    @property
    def accuracy(self) -> float:
        """The share of answers that were right."""
        return float(self.right.float().mean()) if self.n else 0.0

    @property
    def mean_confidence(self) -> float:
        """The mean probability the head gave its answers: equals ``accuracy`` if calibrated."""
        return float(self.confidence.mean()) if self.n else 0.0

    def bins(self, n_bins: int = BINS) -> list[tuple[int, float, float]]:
        """Per equal-width confidence bin (0–1/n, …, up to 1): how many answers, their mean
        confidence and their accuracy. Empty bins are (0, 0, 0)."""
        index = (self.confidence * n_bins).long().clamp(max=n_bins - 1)
        out = []
        for b in range(n_bins):
            mask = index == b
            count = int(mask.sum())
            if not count:
                out.append((0, 0.0, 0.0))
                continue
            conf = float(self.confidence[mask].mean())
            out.append((count, conf, float(self.right[mask].float().mean())))
        return out

    def ece(self, n_bins: int = BINS) -> float:
        """Expected calibration error: the gap between confidence and accuracy in each bin,
        weighted by the bin's share of answers. 0 = calibrated."""
        if not self.n:
            return 0.0
        return sum(count * abs(acc - conf) for count, conf, acc in self.bins(n_bins)) / self.n

    def to_json(self) -> dict[str, Any]:
        """The summary numbers as plain values."""
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "confidence": self.mean_confidence,
            "ece": self.ece(),
        }


def _per_answer(a: _Answered, temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each answer's confidence and whether it is right, with each row's logits divided by its
    example's ``temperature`` (B,): for a value, the product of its tokens' top probabilities,
    and right if every token is."""
    logprobs = (a.logits / temperature[a.example, None]).log_softmax(-1)
    top, said = logprobs.max(-1)
    n = int(a.group.max()) + 1 if a.group.numel() else 0
    total = torch.zeros(n, dtype=top.dtype, device=top.device).index_add_(0, a.group, top)
    wrong = torch.zeros(n, device=top.device).index_add_(0, a.group, (said != a.labels).float())
    return total.exp().cpu(), (wrong == 0).cpu()


def _temperatures(cal: Calibrations, head: str, familiar: torch.Tensor) -> torch.Tensor:
    """(B,) each example's temperature for ``head``: its regime's."""
    return torch.where(familiar, getattr(cal.familiar, head), getattr(cal.new, head))


def reliability(
    model: Eve, task: TaskSet, calibrations: Sequence[Calibrations]
) -> list[dict[str, Reliability]]:
    """Each head's ``Reliability`` on ``task`` under each of ``calibrations`` (one pass), with
    each example in its own regime, as at inference.

    A value's confidence is computed with teacher forcing: the product of each token's top
    probability, reading the gold tokens before it. For a value written right that is exactly
    greedy decoding's probability; for a wrong one, greedy decoding would read its own mistake
    from there on, so it is an estimate.
    """
    parts: list[dict[str, list[tuple[torch.Tensor, torch.Tensor]]]] = [
        {head: [] for head in HEADS} for _ in calibrations
    ]
    for familiar, answered in zip(task.familiar, _answers(model, task.batches), strict=True):
        for cal, out in zip(calibrations, parts, strict=True):
            for head, a in answered.items():
                if a.labels.numel():
                    t = _temperatures(cal, head, familiar).to(a.logits.device)
                    out[head].append(_per_answer(a, t))
    return [
        {
            head: Reliability(
                torch.cat([c for c, _ in pieces]) if pieces else torch.zeros(0),
                torch.cat([r for _, r in pieces]) if pieces else torch.zeros(0, dtype=torch.bool),
            )
            for head, pieces in out.items()
        }
        for out in parts
    ]


def save_calibration(
    path: str | Path, calibrations: Calibrations, fitted_on: dict[str, str]
) -> None:
    """Write the temperatures and the known tools into the checkpoint's meta
    (``Calibrations.from_meta`` reads them); its weights stay as they are."""
    ckpt = load_checkpoint(path)
    meta = ckpt.meta | calibrations.to_meta() | {"calibrated_on": fitted_on}
    save_checkpoint(path, ckpt.model, ckpt.tokenizer, meta, ckpt.train_state)
