import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colibri.config import ModelConfig
from colibri.data.draw import pick
from colibri.data.schema import ASK_VALUE, Arg, Tool

FANOUT = "*"  # {room=*}: one call per option of a required enum ("lights off everywhere")
MAX_CALLS = ModelConfig().n_slots
ROUND_NUMBER_SHARE = 0.5  # people say round numbers more often
NUMBER_WORDS_SHARE = 0.15  # numbers up to 100 written as words: "forty-five"
WHOLE_NUMBER_SHARE = 0.7  # floats: mostly whole numbers, sometimes halves: 21.5
KEEP_OPTIONAL_SHARE = 0.5  # optional segments without a required argument that are kept
SENSIBLE_TRIES = 100  # samples before giving up on a sensible conversion
# Units convert only to units of the same kind: never celsius to km.
UNIT_KINDS = {
    "km": "length",
    "miles": "length",
    "kg": "mass",
    "pounds": "mass",
    "celsius": "temperature",
    "fahrenheit": "temperature",
}

_SLOT = re.compile(r"\{(\w+)(?::([^{}|]*)\|([^{}|]*)|=([^{}]*))?\}")
_OPTIONAL = re.compile(r"\[([^\[\]]*)\]")

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    " fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    """0..999 in words: 7 -> "seven", 45 -> "forty-five", 300 -> "three hundred"."""
    if not 0 <= n <= 999:
        raise ValueError(f"{n} is out of range")
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    rest = n % 100
    return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))


@dataclass(frozen=True)
class Slot:
    arg: str
    true_text: str | None = None  # bool slots: {on:on|off}
    false_text: str | None = None
    fixed: Any = None  # {on=true}: a value with no text
    has_fixed: bool = False


@dataclass(frozen=True)
class Segment:
    optional: bool
    parts: tuple[str | Slot, ...]

    @property
    def args(self) -> set[str]:
        return {p.arg for p in self.parts if isinstance(p, Slot)}


def _segment(text: str, optional: bool) -> Segment:
    parts: list[str | Slot] = []
    pos = 0
    for m in _SLOT.finditer(text):
        parts.append(text[pos : m.start()])
        name, true_text, false_text, fixed = m.groups()
        if fixed is not None:
            parts.append(Slot(name, fixed=_literal(fixed), has_fixed=True))
        else:
            parts.append(Slot(name, true_text, false_text))
        pos = m.end()
    parts.append(text[pos:])
    for p in parts:
        if isinstance(p, str) and any(ch in p for ch in "{}[]"):
            raise ValueError(f"unbalanced or nested brackets in {text!r}")
    return Segment(optional, tuple(p for p in parts if p != ""))


def _literal(text: str) -> Any:
    """{x=true} -> True, {x=50} -> 50, {x=next} -> "next" (bare words are strings)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse(template: str) -> list[Segment]:
    """Split a template into fixed and optional ([...]) segments of text and slots."""
    segments, pos = [], 0
    for m in _OPTIONAL.finditer(template):
        if m.start() > pos:
            segments.append(_segment(template[pos : m.start()], False))
        segments.append(_segment(m.group(1), True))
        pos = m.end()
    if pos < len(template):
        segments.append(_segment(template[pos:], False))
    return segments


def check_template(tool: Tool, template: str, values: Mapping[str, Sequence[str]]) -> list[str]:
    """Problems with a template for ``tool`` (empty list = fine). ``values`` are the str pools."""
    try:
        segments = parse(template)
    except ValueError as e:
        return [str(e)]
    errors = []
    slots = [p for s in segments for p in s.parts if isinstance(p, Slot)]
    names = [s.arg for s in slots]
    if len(set(names)) != len(names):
        errors.append("an argument appears twice")
    args = {a.name: a for a in tool.args}
    if sum(s.has_fixed and s.fixed == FANOUT for s in slots) > 1:
        errors.append("more than one fan-out {name=*}")
    for slot in slots:
        arg = args.get(slot.arg)
        if arg is None:
            errors.append(f"unknown argument {slot.arg!r}")
        elif slot.has_fixed and slot.fixed == FANOUT:
            if arg.type != "enum" or not arg.required:
                errors.append(f"fan-out {{{arg.name}=*}} needs a required enum argument")
            elif len(arg.options) > MAX_CALLS:
                errors.append(f"fan-out over {len(arg.options)} options; at most {MAX_CALLS} calls")
        elif slot.has_fixed:
            if (error := arg.check(slot.fixed)) is not None:
                errors.append(error)
        elif arg.type == "bool" and slot.true_text is None:
            errors.append(f"bool {arg.name} needs {{{arg.name}:yes|no}} or {{{arg.name}=true}}")
        elif arg.type != "bool" and slot.true_text is not None:
            errors.append(f"only bool arguments take {{name:a|b}}, not {arg.name}")
        elif arg.type == "str":
            pool = values.get(arg.name, ())
            if not pool:
                errors.append(f"str argument {arg.name} has no value pool")
            errors += [e for v in pool if (e := arg.check(v)) is not None]
    for arg in tool.args:
        if arg.required and arg.name not in names:
            errors.append(f"required argument {arg.name} is never filled")
    return errors


def _int_value(arg: Arg, rng: np.random.Generator) -> int:
    lo = int(arg.min) if arg.min is not None else 1
    hi = int(arg.max) if arg.max is not None else lo + 99
    if rng.random() < ROUND_NUMBER_SHARE and hi - lo >= 10:
        step = 10 if hi - lo > 100 else 5
        rounds = [v for v in range(lo, hi + 1) if v % step == 0]
        if rounds:
            return int(rng.choice(rounds))
    return int(rng.integers(lo, hi + 1))


def sample_value(
    arg: Arg, slot: Slot, values: Mapping[str, Sequence[str]], rng: np.random.Generator
) -> tuple[Any, str]:
    """A value for ``arg`` and how the utterance says it."""
    if arg.type == "bool":
        v = bool(rng.random() < 0.5)
        return v, slot.true_text if v else slot.false_text
    if arg.type == "enum":
        v = pick(rng, arg.options)
        return v, v
    if arg.type == "str":
        v = pick(rng, values[arg.name])
        return v, v
    if arg.type == "int":
        v = _int_value(arg, rng)
        words = 0 <= v <= 100 and rng.random() < NUMBER_WORDS_SHARE
        return v, number_words(v) if words else str(v)
    lo = arg.min if arg.min is not None else 0.0
    hi = arg.max if arg.max is not None else 100.0
    step = 1 if rng.random() < WHOLE_NUMBER_SHARE else 0.5
    v = float(np.clip(round(rng.uniform(lo, hi) / step) * step, lo, hi))
    return v, arg.format(v)


@dataclass
class Filled:
    text: str
    args: dict[str, Any] = field(default_factory=dict)
    implied: set[str] = field(default_factory=set)  # fixed by the template, not said in the text
    fanout: str | None = None  # an argument to repeat the call for, once per option

    def expand(self, tool: Tool) -> list[dict[str, Any]]:
        """The calls' arguments: one set, or one per option (in schema order) for a fan-out."""
        if self.fanout is None:
            return [dict(self.args)]
        return [{**self.args, self.fanout: option} for option in tool.arg(self.fanout).options]


def sensible(args: Mapping[str, Any]) -> bool:
    """False for a conversion no one asks for: the same value in from_x and to_x (EUR to EUR),
    or units of different kinds (``UNIT_KINDS``)."""
    for name, a in args.items():
        if not name.startswith("from_"):
            continue
        b = args.get("to_" + name.removeprefix("from_"))
        if b is None or ASK_VALUE in (a, b):
            continue
        if a == b or (a in UNIT_KINDS and UNIT_KINDS[a] != UNIT_KINDS.get(b)):
            return False
    return True


def fill(
    tool: Tool,
    template: str,
    values: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
    ask: bool = False,
) -> Filled | None:
    """Fill a template with sampled values.

    Optional segments are kept or dropped at random, except those holding a required argument,
    which are always kept. With ``ask``, exactly one of those is dropped and its required
    argument becomes "?" (the user must be asked). Returns None if ``ask`` is impossible.
    Values that make no ``sensible`` conversion are sampled again.
    """
    for _ in range(SENSIBLE_TRIES):
        filled = _fill(tool, template, values, rng, ask)
        if filled is None or sensible(filled.args):
            return filled
    raise RuntimeError(f"no sensible values for {tool.name}, {template!r}")


def _fill(
    tool: Tool,
    template: str,
    values: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
    ask: bool,
) -> Filled | None:
    segments = parse(template)
    required = {a.name for a in tool.args if a.required}
    droppable = [i for i, s in enumerate(segments) if s.optional and s.args & required]
    if ask and not droppable:
        return None
    drop = pick(rng, droppable) if ask else -1

    args = {a.name: a for a in tool.args}
    out = Filled("")
    words = []
    for i, segment in enumerate(segments):
        keep = (
            not segment.optional
            or (i != drop and bool(segment.args & required))
            or (i != drop and rng.random() < KEEP_OPTIONAL_SHARE)
        )
        if not keep:
            if i == drop:
                for name in segment.args & required:
                    out.args[name] = ASK_VALUE
            continue
        for part in segment.parts:
            if isinstance(part, str):
                words.append(part)
            elif part.has_fixed and part.fixed == FANOUT:
                out.fanout = part.arg
            elif part.has_fixed:
                out.args[part.arg] = part.fixed
                out.implied.add(part.arg)
            else:
                value, text = sample_value(args[part.arg], part, values, rng)
                out.args[part.arg] = value
                words.append(text)
    out.text = tidy("".join(words))
    return out


def tidy(text: str) -> str:
    """Collapse spaces and remove spaces before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.?!%])", r"\1", text)
