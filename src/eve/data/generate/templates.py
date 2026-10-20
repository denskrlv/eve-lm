import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eve.config import ModelConfig
from eve.data.draw import pick
from eve.data.schema import ASK_VALUE, Arg, Tool

FANOUT = "*"  # {room=*}: one call per option of a required enum ("lights off everywhere")
MAX_CALLS = ModelConfig().n_slots
MAX_PROMPT_TOOLS = ModelConfig().max_tools  # a prompt offers 1 to this many tools
DEFAULT_INT_SPAN = 99  # an int without a max goes up to min + this
MIN_ROUNDED_SPAN = 10  # ranges at least this wide may get a round number
WIDE_SPAN, WIDE_ROUND_STEP, NARROW_ROUND_STEP = 100, 10, 5  # round to 10s above 100 wide, else 5s
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


def number_words(number: int) -> str:
    """0..999 in words: 7 -> "seven", 45 -> "forty-five", 300 -> "three hundred"."""
    if not 0 <= number <= 999:
        raise ValueError(f"{number} is out of range")
    if number < 20:
        return _ONES[number]
    if number < 100:
        return _TENS[number // 10] + ("" if number % 10 == 0 else "-" + _ONES[number % 10])
    rest = number % 100
    return _ONES[number // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))


@dataclass(frozen=True)
class Slot:
    """A ``{...}`` slot: the argument it fills, how a bool is said, or a fixed value."""

    arg: str
    true_text: str | None = None  # bool slots: {on:on|off}
    false_text: str | None = None
    fixed: Any = None  # {on=true}: a value with no text
    has_fixed: bool = False


@dataclass(frozen=True)
class Segment:
    """A run of template text and slots; optional when written in ``[...]``."""

    optional: bool
    parts: tuple[str | Slot, ...]

    @property
    def args(self) -> set[str]:
        """The arguments this segment's slots fill."""
        return {p.arg for p in self.parts if isinstance(p, Slot)}


def _segment(text: str, optional: bool) -> Segment:
    parts: list[str | Slot] = []
    pos = 0
    for slot_match in _SLOT.finditer(text):
        parts.append(text[pos : slot_match.start()])
        name, true_text, false_text, fixed = slot_match.groups()
        if fixed is not None:
            parts.append(Slot(name, fixed=_literal(fixed), has_fixed=True))
        else:
            parts.append(Slot(name, true_text, false_text))
        pos = slot_match.end()
    parts.append(text[pos:])
    for part in parts:
        if isinstance(part, str) and any(ch in part for ch in "{}[]"):
            raise ValueError(f"unbalanced or nested brackets in {text!r}")
    return Segment(optional, tuple(part for part in parts if part != ""))


def _literal(text: str) -> Any:
    """{x=true} -> True, {x=50} -> 50, {x=next} -> "next" (bare words are strings)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse(template: str) -> list[Segment]:
    """Split a template into fixed and optional ([...]) segments of text and slots."""
    segments, pos = [], 0
    for optional_match in _OPTIONAL.finditer(template):
        if optional_match.start() > pos:
            segments.append(_segment(template[pos : optional_match.start()], False))
        segments.append(_segment(optional_match.group(1), True))
        pos = optional_match.end()
    if pos < len(template):
        segments.append(_segment(template[pos:], False))
    return segments


def _check_slot(arg: Arg, slot: Slot, values: Mapping[str, Sequence[str]]) -> list[str]:
    """Problems with one slot of a known argument: its fan-out, fixed value, bool words or
    str value pool."""
    if slot.has_fixed and slot.fixed == FANOUT:
        if arg.type != "enum" or not arg.required:
            return [f"fan-out {{{arg.name}=*}} needs a required enum argument"]
        if len(arg.options) > MAX_CALLS:
            return [f"fan-out over {len(arg.options)} options; at most {MAX_CALLS} calls"]
        return []
    if slot.has_fixed:
        error = arg.check(slot.fixed)
        return [] if error is None else [error]
    if arg.type == "bool" and slot.true_text is None:
        return [f"bool {arg.name} needs {{{arg.name}:yes|no}} or {{{arg.name}=true}}"]
    if arg.type != "bool" and slot.true_text is not None:
        return [f"only bool arguments take {{name:a|b}}, not {arg.name}"]
    if arg.type == "str":
        pool = values.get(arg.name, ())
        errors = [] if pool else [f"str argument {arg.name} has no value pool"]
        return errors + [problem for v in pool if (problem := arg.check(v)) is not None]
    return []


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
        else:
            errors += _check_slot(arg, slot, values)
    for arg in tool.args:
        if arg.required and arg.name not in names:
            errors.append(f"required argument {arg.name} is never filled")
    return errors


def _int_value(arg: Arg, rng: np.random.Generator) -> int:
    lo = int(arg.min) if arg.min is not None else 1
    hi = int(arg.max) if arg.max is not None else lo + DEFAULT_INT_SPAN
    if rng.random() < ROUND_NUMBER_SHARE and hi - lo >= MIN_ROUNDED_SPAN:
        step = WIDE_ROUND_STEP if hi - lo > WIDE_SPAN else NARROW_ROUND_STEP
        rounds = [v for v in range(lo, hi + 1) if v % step == 0]
        if rounds:
            return int(rng.choice(rounds))
    return int(rng.integers(lo, hi + 1))


def sample_value(
    arg: Arg, slot: Slot, values: Mapping[str, Sequence[str]], rng: np.random.Generator
) -> tuple[Any, str]:
    """A value for ``arg`` and how the utterance says it."""
    if arg.type == "bool":
        flag = bool(rng.random() < 0.5)
        said = slot.true_text if flag else slot.false_text
        assert said is not None  # check_template: a bool slot names both words
        return flag, said
    if arg.type == "enum":
        option = pick(rng, arg.options)
        return option, option
    if arg.type == "str":
        text = pick(rng, values[arg.name])
        return text, text
    if arg.type == "int":
        number = _int_value(arg, rng)
        words = 0 <= number <= 100 and rng.random() < NUMBER_WORDS_SHARE
        return number, number_words(number) if words else str(number)
    lo = arg.min if arg.min is not None else 0.0
    hi = arg.max if arg.max is not None else 100.0
    step = 1 if rng.random() < WHOLE_NUMBER_SHARE else 0.5
    value = float(np.clip(round(rng.uniform(lo, hi) / step) * step, lo, hi))
    return value, arg.format(value)


@dataclass
class Filled:
    """A filled template: the request text and the call arguments it implies."""

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
    for name, source in args.items():
        if not name.startswith("from_"):
            continue
        target = args.get("to_" + name.removeprefix("from_"))
        if target is None or ASK_VALUE in (source, target):
            continue
        if source == target or (
            source in UNIT_KINDS and UNIT_KINDS[source] != UNIT_KINDS.get(target)
        ):
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


def _keep_segment(
    index: int, segment: Segment, drop: int, required: set[str], rng: np.random.Generator
) -> bool:
    """Whether a segment is in the request: fixed ones always; optional ones that hold a
    required argument unless dropped for ASK (``drop``); other optional ones at random."""
    return (
        not segment.optional
        or (index != drop and bool(segment.args & required))
        or (index != drop and rng.random() < KEEP_OPTIONAL_SHARE)
    )


def _fill_segment(
    segment: Segment,
    args: dict[str, Arg],
    values: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
    out: Filled,
    words: list[str],
) -> None:
    """Add a kept segment's text to ``words`` and its arguments' values to ``out``."""
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


def _fill(
    tool: Tool,
    template: str,
    values: Mapping[str, Sequence[str]],
    rng: np.random.Generator,
    ask: bool,
) -> Filled | None:
    """One attempt at ``fill``: may give values that make no ``sensible`` conversion."""
    segments = parse(template)
    required = {a.name for a in tool.args if a.required}
    droppable = [i for i, s in enumerate(segments) if s.optional and s.args & required]
    if ask and not droppable:
        return None
    drop = pick(rng, droppable) if ask else -1

    args = {a.name: a for a in tool.args}
    out = Filled("")
    words: list[str] = []
    for i, segment in enumerate(segments):
        if _keep_segment(i, segment, drop, required, rng):
            _fill_segment(segment, args, values, rng, out, words)
        elif i == drop:
            for name in segment.args & required:
                out.args[name] = ASK_VALUE
    out.text = tidy("".join(words))
    return out


def tidy(text: str) -> str:
    """Collapse spaces and remove spaces before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.?!%])", r"\1", text)
