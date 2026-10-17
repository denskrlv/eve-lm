from collections.abc import Sequence
from dataclasses import dataclass

import torch

from colibri.config import INTENTS
from colibri.data.render import TaskCollator, TaskExample, render
from colibri.data.schema import Tool
from colibri.model.colibri import Colibri
from colibri.tokenizer.bpe import BPETokenizer
from colibri.train.finetune import ask

STOP = "STOP"


@dataclass
class Answers:
    """What the question heads say about one request, as probabilities."""

    intent: dict[str, float]  # CALL / ASK / UNSUPPORTED / NONE, in INTENTS order
    slots: list[dict[str, float]]  # per "what is call #i?": tool name (or STOP) -> probability

    @property
    def calls(self) -> list[str]:
        """The most likely tool of each slot, up to the first STOP."""
        out = []
        for slot in self.slots:
            best = max(slot, key=slot.get)
            if best == STOP:
                break
            out.append(best)
        return out

    def to_json(self) -> dict:
        return {"intent": self.intent, "calls": self.calls, "slots": self.slots}


@torch.no_grad()
def answer_questions(
    model: Colibri, tokenizer: BPETokenizer, utterance: str, tools: Sequence[Tool]
) -> Answers:
    """One encoder pass and one question pass for a typed request, as at inference (M9).

    The tools stay in the given order. The example's intent is a placeholder: render needs one
    to build the labels, which aren't used here.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    rendered = render(TaskExample(utterance, tuple(tools), "none"), tokenizer, model.cfg)
    batch = {k: v.to(device) for k, v in TaskCollator(model.cfg)([rendered]).items()}
    q = ask(model, batch)
    model.train(was_training)

    names = [t.name for t in tools]
    intent = q.intent[0].softmax(-1).tolist()
    slots = q.slots[0].softmax(-1)[:, [*range(len(names)), model.cfg.max_tools]].tolist()
    return Answers(
        intent=dict(zip(INTENTS, intent, strict=True)),
        slots=[dict(zip([*names, STOP], s, strict=True)) for s in slots],
    )
