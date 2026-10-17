from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from colibri.config import INTENTS, ModelConfig
from colibri.data.generate.build import read_jsonl
from colibri.data.render import IGNORE, TaskCollator, TaskDataset, TaskExample
from colibri.data.schema import Tool
from colibri.model.checkpoint import FRESH, QUESTION_FIELDS, load_checkpoint, load_pretrained
from colibri.model.colibri import Colibri, Questions
from colibri.runtime import seed_everything
from colibri.tokenizer.bpe import BPETokenizer
from colibri.train.loop import TrainConfig, evaluate, train

TASK = "questions"  # meta["task"] of the checkpoints this run writes
HELDOUT_BATCHES = 10


def ask(model: Colibri, batch: dict[str, torch.Tensor]) -> Questions:
    """Encoder + question block only: the decoder isn't needed for the questions."""
    memory = model.encode(batch["enc_ids"], batch["enc_mask"])
    return model.ask_questions(memory, batch["enc_mask"], batch["tool_pos"], batch["tool_mask"])


def _losses(q: Questions, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    slots, labels = q.slots.flatten(0, 1).float(), batch["slots"].flatten()
    if (labels != IGNORE).any():
        slot_loss = F.cross_entropy(slots, labels, ignore_index=IGNORE)
    else:  # every example is an ASK with an unclear tool: no slot labels (CE would give NaN)
        slot_loss = slots.new_zeros(())
    return {"intent": F.cross_entropy(q.intent.float(), batch["intent"]), "slots": slot_loss}


def question_loss(model: Colibri, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Box 8, the classifier fine-tuning loss (book ch. 6.7, ``calc_loss_batch``), for two
    questions at once:

    - ``intent``: cross-entropy of the 4-way CALL / ASK / UNSUPPORTED / NONE answer;
    - ``slots``: the mean cross-entropy of every "what is call #i?" answer over {the prompt's
      tools, STOP}. Slots after the last call should say STOP; NONE and UNSUPPORTED examples
      are all STOP. An ASK whose tool is unclear has no slot labels (IGNORE).

    Each term is a mean over its own labels, and the training loop adds them with weight 1.
    """
    return _losses(ask(model, batch), batch)


def _mean(x: torch.Tensor) -> torch.Tensor:
    """Mean of a 0/1 tensor; 0 if it is empty (a val batch without any such example)."""
    return x.float().mean() if x.numel() else x.new_zeros((), dtype=torch.float)


def question_metrics(model: Colibri, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The val metrics (box 6): both losses plus what they mean for answers.

    - ``intent_acc``: the most likely intent is the right one (book ch. 6.6's accuracy).
    - ``slot_acc``: the share of labelled slots answered right, STOPs included.
    - ``tools_acc``: every slot of an example right: the right tools in the right order, and
      STOP after the last one. Only examples with slot labels count.
    - ``exact_acc``: the intent right and, where there are slot labels, every slot right too.
    - ``call_stop_rate``: the heads disagree: intent CALL, but slot 1 says STOP (no call).
    """
    q = ask(model, batch)
    intent, slots = q.intent.argmax(-1), q.slots.argmax(-1)
    labels = batch["slots"]
    labelled = labels != IGNORE
    rows = labelled.all(-1)  # the renderer labels all of an example's slots or none
    tools_right = (slots == labels).all(-1)
    intent_right = intent == batch["intent"]
    return {
        **_losses(q, batch),
        "intent_acc": _mean(intent_right),
        "slot_acc": _mean((slots == labels)[labelled]),
        "tools_acc": _mean(tools_right[rows]),
        "exact_acc": _mean(intent_right & (tools_right | ~rows)),
        "call_stop_rate": _mean(
            (intent == INTENTS.index("call")) & (slots[:, 0] == model.cfg.max_tools)
        ),
    }


def freeze_body(model: Colibri) -> int:
    """Stage 1 of ch. 6.5: train only the new head and keep the pretrained weights fixed.

    Here the "head" is the whole question block (``questions.*``): its learned queries, its
    two blocks and both heads. The embeddings, encoder and decoder get ``requires_grad=False``,
    so backprop stops at the encoder output and AdamW leaves them alone. Returns the number of
    parameters that still train.
    """
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(FRESH)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_init(path: str | Path, device: str | torch.device = "cpu", **overrides):
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
    model = Colibri(replace(ckpt.model.cfg, **overrides))
    model.load_state_dict(ckpt.model.state_dict())
    ckpt.model = model.to(device)
    return ckpt


def load_examples(path: str | Path, library: dict[str, Tool]) -> list[TaskExample]:
    return [TaskExample.from_json(row, library) for row in read_jsonl(path)]


@torch.no_grad()
def describe(
    model: Colibri, examples: Sequence[TaskExample], data: TaskDataset, indices: Sequence[int]
) -> list[str]:
    """One line per example: what the heads answer, with probabilities, and the gold answer.

    ``turn off the lights → call 97% · set_lights 95%, STOP 99% | gold call · set_lights``
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    rendered = [data[i] for i in indices]
    batch = TaskCollator(model.cfg)(rendered)
    q = ask(model, {k: v.to(device) for k, v in batch.items()})
    model.train(was_training)

    intent_p, slot_p = q.intent.softmax(-1).cpu(), q.slots.softmax(-1).cpu()
    stop = model.cfg.max_tools

    def calls(answers: list[int], tools: list[str]) -> list[str]:
        names = []
        for a in answers:
            names.append("STOP" if a == stop else tools[a])
            if a == stop:
                break
        return names

    lines = []
    for row, (i, r) in enumerate(zip(indices, rendered, strict=True)):
        k = int(intent_p[row].argmax())
        answers = slot_p[row].argmax(-1).tolist()
        said = [
            f"{name} {slot_p[row, s, a]:.0%}"
            for s, (a, name) in enumerate(zip(answers, calls(answers, r.tool_names), strict=False))
        ]
        gold = INTENTS[r.intent]
        if r.slots[0] != IGNORE and r.slots[0] != stop:
            gold += " · " + ", ".join(n for n in calls(r.slots, r.tool_names) if n != "STOP")
        lines.append(
            f"{examples[i].utterance} → {INTENTS[k]} {intent_p[row, k]:.0%} · "
            + ", ".join(said)
            + f" | gold {gold}"
        )
    return lines


def _shown(examples: Sequence[TaskExample]) -> list[int]:
    """The first val example of each intent: the same ones at every eval."""
    first = {}
    for i, e in enumerate(examples):
        first.setdefault(e.intent, i)
    return [first[intent] for intent in INTENTS if intent in first]


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
    overrides: dict | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    sdpa: bool = True,
) -> dict[str, float]:
    """Box 8: fine-tune the question heads on ``task_dir``'s ``train.jsonl``, evaluated on
    ``val.jsonl``, with held-out-tool accuracy from ``heldout-templates.jsonl`` at each eval.

    Start from ``init`` (a pretraining checkpoint or an earlier question run, see ``load_init``)
    or, for the "what does pretraining buy?" experiment (E1), from a random ``scratch`` config
    and the tokenizer at ``tokenizer_path``. ``freeze`` trains only the question block (E2).
    """
    if (init is None) == (scratch is None):
        raise ValueError("give either init (a checkpoint) or scratch (a config)")
    task_dir = Path(task_dir)
    seed_everything(train_cfg.seed)  # the new question block's initialization
    if init is not None:
        ckpt = load_init(init, **(overrides or {}))
        model, tokenizer = ckpt.model, ckpt.tokenizer
    else:
        model = Colibri(replace(scratch, **(overrides or {})))
        tokenizer = BPETokenizer.load(tokenizer_path)
    trainable = freeze_body(model) if freeze else sum(p.numel() for p in model.parameters())
    model.set_sdpa(sdpa)

    cfg = model.cfg
    train_data = TaskDataset(load_examples(task_dir / "train.jsonl", library), tokenizer, cfg)
    val_examples = load_examples(task_dir / "val.jsonl", library)
    val_data = TaskDataset(val_examples, tokenizer, cfg, seed=1)  # tools shuffled, always alike
    heldout = TaskDataset(
        load_examples(task_dir / "heldout-templates.jsonl", library), tokenizer, cfg
    )
    collate = TaskCollator(cfg)
    bs = train_cfg.batch_size
    heldout_batches = [
        collate([heldout[j] for j in range(i * bs, (i + 1) * bs)])
        for i in range(min(HELDOUT_BATCHES, len(heldout) // bs))
    ]
    shown = _shown(val_examples)

    def on_eval(m: Colibri) -> dict:
        lines = describe(m, val_examples, val_data, shown)
        out: dict = {"samples": lines}
        if heldout_batches:
            dev = next(m.parameters()).device
            scores = evaluate(m, question_metrics, heldout_batches, dev, nullcontext)
            scores = {k.removeprefix("val_"): v for k, v in scores.items()}
            out["heldout"] = scores
            lines.append(
                f"held-out tools · intent {scores['intent_acc']:.1%} · "
                f"tools {scores['tools_acc']:.1%} · exact {scores['exact_acc']:.1%}"
            )
        return out

    meta = {"task": TASK, "frozen": freeze, "trainable": trainable, "init": str(init or "scratch")}
    return train(
        model,
        question_loss,
        train_data,
        val_data,
        collate,
        train_cfg,
        out_dir,
        tokenizer,
        device,
        on_eval=on_eval,
        eval_fn=question_metrics,
        resume=resume,
        stop_after=stop_after,
        meta=meta,
    )
