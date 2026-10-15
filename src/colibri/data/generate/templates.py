import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from colibri.data.schema import ASK_VALUE, Arg, Tool

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
    for slot in slots:
        arg = args.get(slot.arg)
        if arg is None:
            errors.append(f"unknown argument {slot.arg!r}")
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
    if rng.random() < 0.5 and hi - lo >= 10:  # people say round numbers more often
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
        v = arg.options[rng.integers(len(arg.options))]
        return v, v
    if arg.type == "str":
        pool = values[arg.name]
        v = pool[rng.integers(len(pool))]
        return v, v
    if arg.type == "int":
        v = _int_value(arg, rng)
        words = 0 <= v <= 100 and rng.random() < 0.15
        return v, number_words(v) if words else str(v)
    lo = arg.min if arg.min is not None else 0.0
    hi = arg.max if arg.max is not None else 100.0
    step = 1 if rng.random() < 0.7 else 0.5  # mostly whole numbers, sometimes halves: 21.5
    v = float(np.clip(round(rng.uniform(lo, hi) / step) * step, lo, hi))
    return v, arg.format(v)


@dataclass
class Filled:
    text: str
    args: dict[str, Any] = field(default_factory=dict)
    implied: set[str] = field(default_factory=set)  # fixed by the template, not said in the text


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
    """
    segments = parse(template)
    required = {a.name for a in tool.args if a.required}
    droppable = [i for i, s in enumerate(segments) if s.optional and s.args & required]
    if ask and not droppable:
        return None
    drop = droppable[rng.integers(len(droppable))] if ask else -1

    args = {a.name: a for a in tool.args}
    out = Filled("")
    words = []
    for i, segment in enumerate(segments):
        keep = (
            not segment.optional
            or (i != drop and bool(segment.args & required))
            or (i != drop and rng.random() < 0.5)
        )
        if not keep:
            if i == drop:
                for name in segment.args & required:
                    out.args[name] = ASK_VALUE
            continue
        for part in segment.parts:
            if isinstance(part, str):
                words.append(part)
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
