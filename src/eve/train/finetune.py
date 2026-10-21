from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eve.config import INTENTS, STOP, ModelConfig
from eve.data.jsonl import read_jsonl
from eve.data.render import IGNORE, Rendered, TaskCollator, TaskDataset, TaskExample
from eve.data.schema import Tool
from eve.model.checkpoint import (
    FRESH,
    QUESTION_FIELDS,
    Checkpoint,
    load_checkpoint,
    load_pretrained,
)
from eve.model.eve import Eve, Output, Questions, ask
from eve.runtime import eval_mode, seed_everything, to_device
from eve.tokenizer.bpe import BPETokenizer
from eve.train.config import TrainConfig
from eve.train.loop import Column, Report, train

TASK = "questions"  # meta["task"] of the checkpoints this run writes
COLUMNS = (
    Column("val loss", "val_loss"),
    Column("val macro", "val.macro_recall"),
    Column("val exact", "val.exact_acc"),
    Column("val values", "val.value_acc"),
    Column("val full", "val.full_acc"),
    Column("held-out macro", "heldout.macro_recall"),
    Column("held-out exact", "heldout.exact_acc"),
)  # the eval table; see score
HIGHLIGHTS = ((r"→ [^|]*", "bold"), (r"\| gold .*", "dim"))  # describe's answers and gold
SCORED = {"val": "val", "heldout": "held-out"}  # on_eval's score groups and their labels
VAL_SEED = 1  # val's tool shuffle: unlike train's (seed 0), and the same at every eval


def _losses(questions: Questions, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    slots, labels = questions.slots.flatten(0, 1).float(), batch["slots"].flatten()
    if (labels != IGNORE).any():
        slot_loss = F.cross_entropy(slots, labels, ignore_index=IGNORE)
    else:  # every example is an ASK with an unclear tool: no slot labels (CE would give NaN)
        slot_loss = slots.new_zeros(())
    return {
        "intent": F.cross_entropy(questions.intent.float(), batch["intent"]),
        "slots": slot_loss,
    }


def question_loss(model: Eve, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Box 8, the classifier fine-tuning loss (book ch. 6.7, ``calc_loss_batch``), for two
    questions at once:

    - ``intent``: cross-entropy of the 4-way CALL / ASK / UNSUPPORTED / NONE answer;
    - ``slots``: the mean cross-entropy of every "what is call #i?" answer over {the prompt's
      tools, STOP}. Slots after the last call should say STOP; NONE and UNSUPPORTED examples
      are all STOP. An ASK whose tool is unclear has no slot labels (IGNORE).

    Each term is a mean over its own labels, and the training loop adds them with weight 1.
    """
    return _losses(ask(model, batch), batch)


def _value_loss(lm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the decoder's value tokens; 0 for a batch without any (only
    UNSUPPORTED, NONE and unclear ASKs, which have no calls)."""
    if not (labels != IGNORE).any():
        return lm.new_zeros((), dtype=torch.float)
    return F.cross_entropy(lm.flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE)


def _forward(model: Eve, batch: dict[str, torch.Tensor]) -> Output:
    """One encoder pass for both the question block and the decoder."""
    out: Output = model(
        batch["enc_ids"],
        batch["enc_mask"],
        batch["dec_ids"],
        batch["dec_mask"],
        batch["tool_pos"],
        batch["tool_mask"],
    )
    return out


def joint_loss(model: Eve, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Box 9, the joint run: ``question_loss`` plus the instruction fine-tuning loss of book
    ch. 7, on the decoder.

    - ``values``: the mean cross-entropy of the decoder's next tokens where it writes an
      argument value (the value's tokens then ``<end>``, or ``<absent>``, or ``<ask>``). The
      structure around them (``<call> name <arg> room <val>``) is copied from the schema, so
      its labels are IGNORE, like the instruction tokens the book masks in ch. 7.5's exercise.

    The decoder reads the gold calls in order (teacher forcing): at inference the slot head
    picks the tools and the decoder only writes their values.
    """
    out = _forward(model, batch)
    assert out.questions is not None and out.lm is not None
    return {**_losses(out.questions, batch), "values": _value_loss(out.lm, batch["dec_labels"])}


def _mean(flags: torch.Tensor) -> torch.Tensor:
    """Mean of a 0/1 tensor; 0 if it is empty (a val batch without any such example)."""
    return flags.float().mean() if flags.numel() else flags.new_zeros((), dtype=torch.float)


def value_hits(lm: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """How many argument values each example has and how many the decoder writes right.

    A value is a run of labelled decoder positions: the schema's structure tokens (IGNORE)
    separate one from the next. It is right if every token of it is the most likely one, up to
    and including its ``<end>``, so this is what greedy decoding would write given the gold
    tools and the values before it. Returns two (B,) counts.
    """
    labelled = labels != IGNORE
    starts = labelled & ~F.pad(labelled[:, :-1], (1, 0), value=False)
    wrong = labelled & (lm.argmax(-1) != labels)
    segment = starts.long().cumsum(-1) * wrong  # the value each wrong token is in, else 0
    misses = torch.zeros(labels.shape[0], labels.shape[1] + 1, device=labels.device)
    misses.scatter_add_(1, segment, wrong.float())
    values = starts.sum(-1)
    return values, values - (misses[:, 1:] > 0).sum(-1)


@dataclass(frozen=True)
class _Graded:
    """The heads' answers for a batch, checked against its labels."""

    intent: torch.Tensor  # (B,) the most likely intent
    slots: torch.Tensor  # (B, n_slots) the most likely answer per slot
    intent_right: torch.Tensor  # (B,)
    labelled: torch.Tensor  # (B, n_slots) slots with a label
    has_slots: torch.Tensor  # (B,) the renderer labels all of an example's slots or none
    tools_right: torch.Tensor  # (B,) every slot right: the right tools in order, then STOP
    values: torch.Tensor  # (B,) argument values in the gold calls; 0 without the decoder
    values_right: torch.Tensor  # (B,) of those, the ones the decoder writes right

    @property
    def exact(self) -> torch.Tensor:
        """(B,) the intent right and, where there are slot labels, every slot right too."""
        return self.intent_right & (self.tools_right | ~self.has_slots)


def _grade(
    questions: Questions, batch: dict[str, torch.Tensor], lm: torch.Tensor | None = None
) -> _Graded:
    intent, slots = questions.intent.argmax(-1), questions.slots.argmax(-1)
    labelled = batch["slots"] != IGNORE
    if lm is None:
        values = values_right = intent.new_zeros(intent.shape)
    else:
        values, values_right = value_hits(lm, batch["dec_labels"])
    return _Graded(
        intent=intent,
        slots=slots,
        intent_right=intent == batch["intent"],
        labelled=labelled,
        has_slots=labelled.all(-1),
        tools_right=(slots == batch["slots"]).all(-1),
        values=values,
        values_right=values_right,
    )


def _ratio(right: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    """``right / total`` as a float; 0 if ``total`` is 0."""
    return right.sum().float() / total.sum().clamp(min=1)


def question_metrics(model: Eve, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The val metrics (box 6): both losses and their sum ``loss``, plus what they mean for
    answers.

    - ``intent_acc``: the most likely intent is the right one (book ch. 6.6's accuracy).
    - ``slot_acc``: the share of labelled slots answered right, STOPs included.
    - ``tools_acc``: every slot of an example right: the right tools in the right order, and
      STOP after the last one. Only examples with slot labels count.
    - ``exact_acc``: the intent right and, where there are slot labels, every slot right too.
    - ``call_stop_rate``: the heads disagree: intent CALL, but slot 1 says STOP (no call).
    """
    questions = ask(model, batch)
    return _question_metrics(model, _grade(questions, batch), _losses(questions, batch), batch)


def _question_metrics(
    model: Eve,
    graded: _Graded,
    losses: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        **losses,
        "loss": torch.stack(list(losses.values())).sum(),
        "intent_acc": _mean(graded.intent_right),
        "slot_acc": _mean((graded.slots == batch["slots"])[graded.labelled]),
        "tools_acc": _mean(graded.tools_right[graded.has_slots]),
        "exact_acc": _mean(graded.exact),
        "call_stop_rate": _mean(
            (graded.intent == INTENTS.index("call")) & (graded.slots[:, 0] == model.cfg.max_tools)
        ),
    }


def joint_metrics(model: Eve, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``question_metrics`` for the joint run: the ``values`` loss is in ``loss`` too, plus

    - ``value_acc``: the share of argument values the decoder writes right (``value_hits``),
      ``<absent>`` and ``<ask>`` included, given the gold tools.
    - ``full_acc``: exact, and every value of the example right too: the whole answer.
    """
    out = _forward(model, batch)
    assert out.questions is not None and out.lm is not None
    graded = _grade(out.questions, batch, out.lm)
    losses = _losses(out.questions, batch) | {"values": _value_loss(out.lm, batch["dec_labels"])}
    return _question_metrics(model, graded, losses, batch) | {
        "value_acc": _ratio(graded.values_right, graded.values),
        "full_acc": _mean(graded.exact & (graded.values_right == graded.values)),
    }


@dataclass(frozen=True)
class Outcome:
    """One example's gold and answered intent, whether its slots were right, and how many of
    its argument values the decoder wrote right."""

    gold: str
    said: str
    has_slots: bool  # the example has slot labels (an ASK with an unclear tool has none)
    tools_ok: bool  # every slot right: the right tools in order, then STOP
    values: int = 0  # argument values in the gold calls (0 if the decoder wasn't scored)
    values_right: int = 0


@dataclass
class Tally:
    """Answers counted over a whole set: how many examples of each intent there are (``seen``)
    and how many got it (``right``), the same for ``tools``, ``exact``, ``values`` and
    ``full``, and which intent each gold intent was answered with (``answered[gold, said]``)."""

    seen: Counter[str] = field(default_factory=Counter)
    right: Counter[str] = field(default_factory=Counter)
    answered: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, outcome: Outcome) -> None:
        """Count one example."""
        gold, intent_ok, has_slots = outcome.gold, outcome.gold == outcome.said, outcome.has_slots
        exact = intent_ok and (outcome.tools_ok or not has_slots)
        self.seen[gold] += 1
        self.right[gold] += intent_ok
        self.answered[gold, outcome.said] += 1
        self.seen["tools"] += has_slots
        self.right["tools"] += has_slots and outcome.tools_ok
        self.right["exact"] += exact
        self.seen["values"] += outcome.values
        self.right["values"] += outcome.values_right
        self.right["full"] += exact and outcome.values_right == outcome.values

    def scores(self) -> dict[str, float]:
        """The recall per intent and its mean, ``tools_acc`` and ``exact_acc``, and, if the
        decoder was scored, ``value_acc`` and ``full_acc`` (see ``score``)."""
        recalls = {f"recall_{i}": self.right[i] / self.seen[i] for i in INTENTS if self.seen[i]}
        total = sum(self.seen[i] for i in INTENTS)
        out = {
            "macro_recall": sum(recalls.values()) / len(recalls),
            **recalls,
            "tools_acc": self.right["tools"] / max(1, self.seen["tools"]),
            "exact_acc": self.right["exact"] / total,
        }
        if self.seen["values"]:
            out["value_acc"] = self.right["values"] / self.seen["values"]
            out["full_acc"] = self.right["full"] / total
        return out


@torch.no_grad()
def outcomes(
    model: Eve, batches: Sequence[dict[str, torch.Tensor]], values: bool = True
) -> list[Outcome]:
    """The heads' answers for every example of ``batches``, in order, checked against gold.
    With ``values``, the decoder's argument values are checked too (``value_hits``)."""
    out = []
    with eval_mode(model) as device:
        for batch in batches:
            batch = to_device(batch, device)
            if values:
                answers = _forward(model, batch)
                assert answers.questions is not None
                graded = _grade(answers.questions, batch, answers.lm)
            else:
                graded = _grade(ask(model, batch), batch)
            columns = (
                batch["intent"],
                graded.intent,
                graded.has_slots,
                graded.tools_right,
                graded.values,
                graded.values_right,
            )
            for gold, said, *rest in zip(*(c.tolist() for c in columns), strict=True):
                out.append(Outcome(INTENTS[gold], INTENTS[said], *rest))
    return out


def tally(answers: Sequence[Outcome]) -> Tally:
    """``answers`` counted in one ``Tally``."""
    out = Tally()
    for outcome in answers:
        out.add(outcome)
    return out


def score(
    model: Eve, batches: Sequence[dict[str, torch.Tensor]], values: bool = True
) -> dict[str, float]:
    """Accuracy counted over all of ``batches`` at once, not averaged per batch, so rare
    classes count right. A recall per intent says how often its examples get that answer:
    val is 84% CALL, so answering CALL every time already gets 84% intent accuracy, but a
    recall of 0 for ASK, UNSUPPORTED and NONE. ``macro_recall`` is the mean of the recalls.
    With ``values``, also ``value_acc`` and ``full_acc`` (see ``joint_metrics``).
    """
    return tally(outcomes(model, batches, values)).scores()


def score_line(scores: dict[str, float]) -> str:
    """``macro recall 52.0% (call 90% · ask 10% · …) · tools 51.0% · exact 40.0%``, then
    ``· values 80.0% · full 35.0%`` if the decoder was scored."""
    recalls = " · ".join(
        f"{k.removeprefix('recall_')} {v:.0%}" for k, v in scores.items() if k.startswith("recall_")
    )
    line = (
        f"macro recall {scores['macro_recall']:.1%} ({recalls}) · "
        f"tools {scores['tools_acc']:.1%} · exact {scores['exact_acc']:.1%}"
    )
    if "value_acc" in scores:
        line += f" · values {scores['value_acc']:.1%} · full {scores['full_acc']:.1%}"
    return line


def summary(results: dict[str, Any]) -> list[tuple[str, str]]:
    """The run's last scores, one labelled line per scored set, for the end of a run."""
    return [(label, score_line(results[key])) for key, label in SCORED.items() if key in results]


def freeze_body(model: Eve) -> int:
    """Stage 1 of ch. 6.5: train only the new head and keep the pretrained weights fixed.

    Here the "head" is the whole question block (``questions.*``): its learned queries, its
    two blocks and both heads. The embeddings, encoder and decoder get ``requires_grad=False``,
    so backprop stops at the encoder output and AdamW leaves them alone. Returns the number of
    parameters that still train.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(FRESH)
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def load_init(path: str | Path, device: str | torch.device = "cpu", **overrides: Any) -> Checkpoint:
    """The model to fine-tune, from either kind of checkpoint:

    - a pretraining checkpoint (box 7): ``load_pretrained``, so the question block starts new;
    - a checkpoint of an earlier question run (``meta["task"] == "questions"``), e.g. stage 1
      with the body frozen: everything is loaded, the trained question block too.

    ``overrides`` may change the question block's config, as in ``load_pretrained``; for an
    earlier question run only ``dropout`` (other changes would leave weights without a home).
    """
    if bad := set(overrides) - QUESTION_FIELDS:
        raise ValueError(f"can't change {sorted(bad)} of a pretrained body")
    ckpt = load_checkpoint(path)
    if ckpt.meta.get("task") != TASK:
        return load_pretrained(path, device, **overrides)
    model = Eve(replace(ckpt.model.cfg, **overrides))
    model.load_state_dict(ckpt.model.state_dict())
    ckpt.model = model.to(device)
    return ckpt


def load_examples(path: str | Path, library: dict[str, Tool]) -> list[TaskExample]:
    """The examples of a task .jsonl file, with tools looked up in ``library``."""
    return [TaskExample.from_json(row, library) for row in read_jsonl(path)]


def _calls(answers: list[int], tools: list[str], stop: int) -> list[str]:
    """Slot answers as tool names, up to and including the first STOP."""
    names = []
    for answer in answers:
        names.append(STOP if answer == stop else tools[answer])
        if answer == stop:
            break
    return names


def _describe_line(
    utterance: str,
    intent_probs: torch.Tensor,
    slot_probs: torch.Tensor,
    item: Rendered,
    stop: int,
) -> str:
    """One example's line for ``describe``, from its intent (n_intents,) and slot
    (n_slots, max_tools + 1) probabilities."""
    best = int(intent_probs.argmax())
    answers = slot_probs.argmax(-1).tolist()
    said = [
        f"{name} {slot_probs[s, a]:.0%}"
        for s, (a, name) in enumerate(
            zip(answers, _calls(answers, item.tool_names, stop), strict=False)
        )
    ]
    gold = INTENTS[item.intent]
    if item.slots[0] != IGNORE and item.slots[0] != stop:
        gold += " · " + ", ".join(n for n in _calls(item.slots, item.tool_names, stop) if n != STOP)
    return (
        f"{utterance} → {INTENTS[best]} {intent_probs[best]:.0%} · "
        + ", ".join(said)
        + f" | gold {gold}"
    )


@torch.no_grad()
def describe(
    model: Eve, examples: Sequence[TaskExample], data: TaskDataset, indices: Sequence[int]
) -> list[str]:
    """One line per example: what the heads answer, with probabilities, and the gold answer.

    ``turn off the lights → call 97% · set_lights 95%, STOP 99% | gold call · set_lights``
    """
    rendered = [data[i] for i in indices]
    with eval_mode(model) as device:
        questions = ask(model, to_device(TaskCollator(model.cfg)(rendered), device))

    intent_p, slot_p = questions.intent.softmax(-1).cpu(), questions.slots.softmax(-1).cpu()
    stop = model.cfg.max_tools
    return [
        _describe_line(examples[i].utterance, intent_p[row], slot_p[row], item, stop)
        for row, (i, item) in enumerate(zip(indices, rendered, strict=True))
    ]


def _shown(examples: Sequence[TaskExample]) -> list[int]:
    """The first val example of each intent: the same ones at every eval."""
    first: dict[str, int] = {}
    for i, example in enumerate(examples):
        first.setdefault(example.intent, i)
    return [first[intent] for intent in INTENTS if intent in first]


def _start_model(
    init: str | Path | None,
    scratch: ModelConfig | None,
    tokenizer_path: str | Path | None,
    overrides: dict[str, Any],
) -> tuple[Eve, BPETokenizer]:
    """The model to fine-tune and its tokenizer: from the ``init`` checkpoint, or a random model
    with the ``scratch`` config and the tokenizer at ``tokenizer_path``."""
    if init is not None:
        ckpt = load_init(init, **overrides)
        return ckpt.model, ckpt.tokenizer
    assert scratch is not None  # finetune checks: exactly one of init and scratch
    if tokenizer_path is None:
        raise ValueError("scratch needs tokenizer_path")
    model = Eve(replace(scratch, **overrides))
    return model, BPETokenizer.load(tokenizer_path)


def in_batches(
    data: TaskDataset, collate: TaskCollator, size: int
) -> list[dict[str, torch.Tensor]]:
    """All of ``data`` in order, in batches of ``size`` (the last one may be smaller)."""
    return [
        collate([data[j] for j in range(i, min(i + size, len(data)))])
        for i in range(0, len(data), size)
    ]


def _eval_report(
    val_examples: Sequence[TaskExample],
    val_data: TaskDataset,
    heldout: TaskDataset,
    collate: TaskCollator,
    batch_size: int,
    values: bool,
) -> Callable[[Eve], dict[str, Any]]:
    """``on_eval`` for fine-tuning: the answers for the first val example of each intent, and
    the val and held-out-tool scores, with the decoder's values if ``values``."""
    scored = {
        "val": in_batches(val_data, collate, batch_size),
        "heldout": in_batches(heldout, collate, batch_size),
    }
    shown = _shown(val_examples)

    def on_eval(model: Eve) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": describe(model, val_examples, val_data, shown)}
        out |= {key: score(model, data, values) for key, data in scored.items() if data}
        return out

    return on_eval


def _task_data(
    task_dir: Path, library: dict[str, Tool], tokenizer: BPETokenizer, cfg: ModelConfig
) -> tuple[TaskDataset, list[TaskExample], TaskDataset, TaskDataset]:
    """The train set, the val examples and set, and the held-out-tool set of ``task_dir``."""
    train_data = TaskDataset(load_examples(task_dir / "train.jsonl", library), tokenizer, cfg)
    val_examples = load_examples(task_dir / "val.jsonl", library)
    val_data = TaskDataset(val_examples, tokenizer, cfg, seed=VAL_SEED)
    heldout = TaskDataset(
        load_examples(task_dir / "heldout-templates.jsonl", library), tokenizer, cfg
    )
    return train_data, val_examples, val_data, heldout


def finetune(
    train_cfg: TrainConfig,
    task_dir: str | Path,
    out_dir: str | Path,
    device: str | torch.device,
    library: dict[str, Tool],
    init: str | Path | None = None,
    scratch: ModelConfig | None = None,
    tokenizer_path: str | Path | None = None,
    freeze: bool = False,
    overrides: dict[str, Any] | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    sdpa: bool = True,
) -> dict[str, float]:
    """Boxes 8 and 9: fine-tune the question heads and the decoder's argument values together
    (``joint_loss``) on ``task_dir``'s ``train.jsonl``, evaluated on ``val.jsonl``, with
    held-out-tool accuracy from ``heldout-templates.jsonl`` at each eval.

    Start from ``init`` (a pretraining checkpoint or an earlier question run, see ``load_init``)
    or, for the "what does pretraining buy?" experiment (E1), from a random ``scratch`` config
    and the tokenizer at ``tokenizer_path``. ``freeze`` trains only the question block (E2), so
    it trains on ``question_loss`` alone: the frozen decoder would learn nothing from its loss.
    """
    if (init is None) == (scratch is None):
        raise ValueError("give either init (a checkpoint) or scratch (a config)")
    task_dir = Path(task_dir)
    seed_everything(train_cfg.seed)  # the new question block's initialization
    model, tokenizer = _start_model(init, scratch, tokenizer_path, overrides or {})
    trainable = freeze_body(model) if freeze else sum(p.numel() for p in model.parameters())
    model.set_sdpa(sdpa)

    cfg = model.cfg
    train_data, val_examples, val_data, heldout = _task_data(task_dir, library, tokenizer, cfg)
    collate = TaskCollator(cfg)
    joint = not freeze
    on_eval = _eval_report(val_examples, val_data, heldout, collate, train_cfg.batch_size, joint)
    meta = {
        "task": TASK,
        "frozen": freeze,
        "joint": joint,  # the decoder was trained to write argument values too
        "trainable": trainable,
        "init": str(init or "scratch"),
    }
    return train(
        model,
        joint_loss if joint else question_loss,
        train_data,
        val_data,
        collate,
        train_cfg,
        out_dir,
        tokenizer,
        device,
        report=Report(COLUMNS, on_eval, summary, HIGHLIGHTS),
        eval_fn=joint_metrics if joint else question_metrics,
        resume=resume,
        stop_after=stop_after,
        meta=meta,
    )
