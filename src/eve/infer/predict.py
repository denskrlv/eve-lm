import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from eve.config import INTENTS, STOP
from eve.data.render import TaskCollator, TaskExample, render
from eve.data.schema import Arg, SchemaError, Tool
from eve.model.eve import Eve
from eve.runtime import eval_mode, to_device
from eve.tokenizer.bpe import ABSENT, ARG, ASK, BYTE0, CALL, DEC, END, VAL, BPETokenizer

MAX_VALUE_TOKENS = 32  # a value longer than this is cut off and marked invalid
KINDS = ("value", "absent", "ask", "invalid")  # what the decoder can write for an argument


@dataclass(frozen=True)
class Value:
    """What the decoder wrote for one argument."""

    kind: str  # one of KINDS
    value: Any = None  # the parsed value; for "invalid", the text as written
    prob: float = 1.0  # probability of the written tokens, the <end>/<absent>/<ask> included
    error: str | None = None  # why an "invalid" value isn't one

    def to_json(self) -> dict[str, Any]:
        """The value as plain values, without the fields that don't apply."""
        out: dict[str, Any] = {"kind": self.kind, "prob": self.prob}
        if self.kind in ("value", "invalid"):
            out["value"] = self.value
        return out | ({"error": self.error} if self.error else {})


@dataclass(frozen=True)
class WrittenCall:
    """One call the slot head picked, with the values the decoder wrote for its arguments."""

    tool: str
    prob: float  # the slot head's probability of this tool
    values: dict[str, Value]  # argument name -> what was written, in schema order

    @property
    def args(self) -> dict[str, Any]:
        """The arguments that have a value: the call to make."""
        return {name: v.value for name, v in self.values.items() if v.kind == "value"}

    @property
    def missing(self) -> list[str]:
        """The required arguments the user didn't give: what to ask them for."""
        return [name for name, v in self.values.items() if v.kind == "ask"]

    @property
    def invalid(self) -> list[str]:
        """The arguments whose written value doesn't fit the schema."""
        return [name for name, v in self.values.items() if v.kind == "invalid"]

    def to_json(self) -> dict[str, Any]:
        """``{"tool", "args", "missing", "prob", "values"}``: the call, then the details."""
        return {
            "tool": self.tool,
            "args": self.args,
            "missing": self.missing,
            "prob": self.prob,
            "values": {name: v.to_json() for name, v in self.values.items()},
        }


@dataclass
class Answers:
    """What the question heads say about one request, as probabilities, and the calls with
    their argument values (M8) if they were written."""

    intent: dict[str, float]  # CALL / ASK / UNSUPPORTED / NONE, in INTENTS order
    slots: list[dict[str, float]]  # per "what is call #i?": tool name (or STOP) -> probability
    written: list[WrittenCall] = field(default_factory=list)  # one per name in ``calls``

    @property
    def calls(self) -> list[str]:
        """The most likely tool of each slot, up to the first STOP."""
        out = []
        for slot in self.slots:
            best = max(slot, key=slot.__getitem__)
            if best == STOP:
                break
            out.append(best)
        return out

    def to_json(self) -> dict[str, Any]:
        """The answers as plain values: the whole calls if they were written, else the most
        likely tools."""
        calls = [c.to_json() for c in self.written] if self.written else self.calls
        return {"intent": self.intent, "calls": calls, "slots": self.slots}


def _allowed(arg: Arg, first: bool, vocab_size: int, n_tokens: int) -> torch.Tensor:
    """(vocab_size,) True where the decoder may write that token next in ``arg``'s value.

    Text tokens always (those the tokenizer has); ``<end>`` after at least one of them; as the
    first token, ``<ask>`` for a required argument and ``<absent>`` for an optional one, as
    the renderer writes them. Structure tokens and sentinels never.
    """
    allowed = torch.zeros(vocab_size, dtype=torch.bool)
    allowed[BYTE0:n_tokens] = True
    if first:
        allowed[ASK if arg.required else ABSENT] = True
    else:
        allowed[END] = True
    return allowed


def _write_value(
    model: Eve,
    memory: torch.Tensor,
    enc_mask: torch.Tensor,
    seq: list[int],
    arg: Arg,
    n_tokens: int,
) -> tuple[list[int], float]:
    """Greedy decoding of one value after ``seq``, which ends in ``<val>``: the tokens up to
    and including ``<end>``, ``<absent>`` or ``<ask>`` (or cut off), and their probability."""
    ids: list[int] = []
    logprob = 0.0
    limit = min(MAX_VALUE_TOKENS + 1, model.cfg.max_decoder_len - len(seq))
    masks = {
        first: _allowed(arg, first, model.cfg.vocab_size, n_tokens).to(memory.device)
        for first in (True, False)
    }
    while len(ids) < limit:
        dec = torch.tensor([seq + ids], device=memory.device)
        logits = model.decode(dec, memory, enc_mask)[0, -1].float()
        logprobs = logits.masked_fill(~masks[not ids], float("-inf")).log_softmax(-1)
        token = int(logprobs.argmax())
        logprob += float(logprobs[token])
        ids.append(token)
        if token in (END, ABSENT, ASK):
            break
    return ids, math.exp(logprob)


def _read(arg: Arg, ids: list[int], prob: float, tokenizer: BPETokenizer) -> Value:
    """The written tokens of one value as a ``Value``."""
    if ids and ids[-1] == ABSENT:
        return Value("absent", prob=prob)
    if ids and ids[-1] == ASK:
        return Value("ask", prob=prob)
    ended = bool(ids) and ids[-1] == END
    text = tokenizer.decode(ids[:-1] if ended else ids).strip()
    if not ended:
        return Value("invalid", text, prob, f"{arg.name}: no <end> after the value")
    try:
        return Value("value", arg.parse(text), prob)
    except SchemaError as e:
        return Value("invalid", text, prob, str(e))


@torch.no_grad()
def write_values(
    model: Eve,
    tokenizer: BPETokenizer,
    memory: torch.Tensor,
    enc_mask: torch.Tensor,
    tools: Sequence[Tool],
) -> list[dict[str, Value]]:
    """The decoder's values for a call to each of ``tools``, in order (M8).

    The decoder sequence is built as in training (``render._decoder``): the structure is copied
    from the schema, ``<call> set_lights <arg> room <val>``, and the decoder writes only each
    value, by greedy decoding up to ``<end>``, ``<absent>`` or ``<ask>``. Each value reads the
    calls and values before it. ``memory`` and ``enc_mask`` are one example's (batch of 1).
    """
    seq, out = [DEC], []
    for tool in tools:
        seq += [CALL, *tokenizer.encode(" " + tool.name)]
        values = {}
        for arg in tool.args:
            seq += [ARG, *tokenizer.encode(" " + arg.name), VAL]
            ids, prob = _write_value(model, memory, enc_mask, seq, arg, tokenizer.vocab_size)
            seq += ids
            values[arg.name] = _read(arg, ids, prob, tokenizer)
        out.append(values)
    return out


@torch.no_grad()
def answer_questions(
    model: Eve,
    tokenizer: BPETokenizer,
    utterance: str,
    tools: Sequence[Tool],
    values: bool = False,
) -> Answers:
    """One encoder pass and one question pass for a typed request, as at inference (M9). With
    ``values``, the decoder then writes the arguments of each call the slot head picked (M8):
    only for a model from the joint run, whose decoder learned to.

    The tools stay in the given order. The example's intent is a placeholder: render needs one
    to build the labels, which aren't used here.
    """
    rendered = render(TaskExample(utterance, tuple(tools), "none"), tokenizer, model.cfg)
    names = [t.name for t in tools]
    with eval_mode(model) as device:
        batch = to_device(TaskCollator(model.cfg)([rendered]), device)
        memory = model.encode(batch["enc_ids"], batch["enc_mask"])
        questions = model.ask_questions(
            memory, batch["enc_mask"], batch["tool_pos"], batch["tool_mask"]
        )
        intent = questions.intent[0].softmax(-1).tolist()
        slots = questions.slots[0].softmax(-1)[:, [*range(len(names)), model.cfg.max_tools]]
        answers = Answers(
            intent=dict(zip(INTENTS, intent, strict=True)),
            slots=[dict(zip([*names, STOP], s, strict=True)) for s in slots.tolist()],
        )
        if values:
            by_name = {t.name: t for t in tools}
            called = [by_name[name] for name in answers.calls]
            written = write_values(model, tokenizer, memory, batch["enc_mask"], called)
            answers.written = [
                WrittenCall(name, answers.slots[i][name], vals)
                for i, (name, vals) in enumerate(zip(answers.calls, written, strict=True))
            ]
    return answers
