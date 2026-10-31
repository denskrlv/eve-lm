"""What every TinyTools converter shares: the travel filter, the eval-set dedupe, number words,
and the document formats (dialogues, "ways to say it" lists, tool cards, scenes)."""

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from eve import defaults
from eve.data.draw import pick, shuffled
from eve.data.jsonl import read_jsonl

# The held-out travel tools (search_flights, book_hotel, get_flight_status, convert_currency) stay
# unseen: a request or document that touches them is dropped. Narrower than fine-tuning's
# HELD_OUT, which also drops "trip", "travel" and "convert": pretraining text may mention a trip to
# the zoo or converting sunlight into energy.
TRAVEL = re.compile(
    r"\b(flights?|airports?|airlines?|airfares?|plane tickets?|boarding pass(es)?|hotels?|motels?|"
    r"currenc(y|ies)|exchange rates?|search_flights|book_hotel|get_flight_status|"
    r"convert_currency)\b",
    re.IGNORECASE,
)
# Words that don't make two requests different: "olly" is MASSIVE's and HWU64's wake word.
FILLER = frozenset("please olly hey ok okay the a an can could would you me i to for".split())
MIN_KEY_WORDS = 2  # words (not numbers): a shorter request ("lights") matches too much


class Doc(NamedTuple):
    """A TinyTools document and its kind: dialogue, how (things work), ways, card or scene."""

    text: str
    kind: str


def request_key(text: str) -> str:
    """A request with case, punctuation, numbers and filler words removed, so near-identical
    requests ("Please set an alarm for 7." / "set alarm for 8") get the same key."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    words = ["#" if w.isdigit() else w for w in words if w not in FILLER]
    return " ".join(words) if sum(w != "#" for w in words) >= MIN_KEY_WORDS else ""


@dataclass
class EvalSet:
    """The requests of the eval sets, by ``request_key``. Chitchat (intent NONE: "hi there",
    "sounds good") is left out: it is all over web text anyway, and matching it would drop
    whole dialogues for a greeting."""

    keys: frozenset[str] = frozenset()
    files: tuple[str, ...] = ()

    @classmethod
    def load(cls, evals_dir: str | Path = defaults.EVALS_DIR) -> "EvalSet":
        """Every set in ``evals_dir``: TinyTools must contain none of them."""
        keys: set[str] = set()
        files = []
        for path in sorted(Path(evals_dir).glob("*.jsonl")):
            keys.update(
                request_key(row["utterance"])
                for row in read_jsonl(path)
                if row.get("intent") != "none"
            )
            files.append(path.name)
        keys.discard("")
        return cls(frozenset(keys), tuple(files))

    def __contains__(self, text: str) -> bool:
        key = request_key(text)
        return bool(key) and key in self.keys


@dataclass
class Context:
    """What a converter needs besides its raw files: randomness, the filters and word lists, and
    a counter of what it dropped and why."""

    rng: np.random.Generator
    evals: EvalSet = field(default_factory=EvalSet)
    names: Sequence[str] = ("Sam",)
    nouns: frozenset[str] = frozenset()
    person_nouns: frozenset[str] = frozenset()  # man, child, teacher...: a subset of nouns
    dropped: Counter[str] = field(default_factory=Counter)

    def keep(self, texts: str | Iterable[str]) -> bool:
        """The requests (or turns) touch no travel topic and appear in no eval set; else count
        why they don't and return False."""
        for text in [texts] if isinstance(texts, str) else texts:
            if TRAVEL.search(text):
                self.dropped["travel topic"] += 1
                return False
            if text in self.evals:
                self.dropped["in an eval set"] += 1
                return False
        return True

    def drop(self, reason: str, count: int = 1) -> None:
        self.dropped[reason] += count

    def name(self) -> str:
        return pick(self.rng, self.names)


def read_words(path: str | Path) -> list[str]:
    """The lines of a word list, without comments and blank lines."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def read_sections(path: str | Path) -> dict[str, list[str]]:
    """A word list's words by section: the comment line above them ("# people and family")."""
    sections: dict[str, list[str]] = {}
    section = ""
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
        elif line.strip():
            sections.setdefault(section, []).append(line.strip())
    return sections


_IRREGULAR = {
    "child": "children", "man": "men", "woman": "women", "person": "people", "foot": "feet",
    "tooth": "teeth", "mouse": "mice", "goose": "geese", "knife": "knives", "leaf": "leaves",
    "wife": "wives", "life": "lives", "shelf": "shelves", "loaf": "loaves", "half": "halves",
}  # fmt: skip


def plural(noun: str) -> str:
    """kettle -> kettles, battery -> batteries, glass -> glasses, child -> children."""
    if noun in _IRREGULAR:
        return _IRREGULAR[noun]
    if noun.endswith("y") and noun[-2:-1] not in tuple("aeiou"):
        return noun[:-1] + "ies"
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    return noun + "s"


def clean(text: str | None) -> str:
    """Whitespace collapsed to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def humanize(label: str) -> str:
    """An intent label as words: card_arrival / AddToPlaylist / HassTurnOn -> "turn on"."""
    label = re.sub(r"^Hass(?=[A-Z])", "", label)
    label = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
    return re.sub(r"[_\-\s]+", " ", label).strip().lower()


def chunks[T](
    items: Sequence[T], rng: np.random.Generator, low: int, high: int
) -> Iterator[list[T]]:
    """``items`` cut into consecutive runs of ``low``..``high`` items (the last may be shorter)."""
    start = 0
    while start < len(items):
        size = int(rng.integers(low, high + 1))
        yield list(items[start : start + size])
        start += size


# ---- numbers in words -----------------------------------------------------------------------

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# A whole number standing alone: not part of a decimal, a time, a price, a percentage or a word.
_NUMBER = re.compile(r"(?<![\w$£€.,:/-])\d{1,6}(?![\w%]|[.,:/]\d)")


def number_words(n: int) -> str:
    """0..999,999 in words: 21 -> "twenty-one", 305 -> "three hundred and five"."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + number_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        joiner = "" if rest == 0 else (" and " if rest < 100 else " ")
        return number_words(n // 1000) + " thousand" + joiner + (number_words(rest) if rest else "")
    raise ValueError(f"{n} is too large")


def spell_numbers(text: str) -> str:
    """Every whole number below a million that stands alone, written in words."""
    return _NUMBER.sub(lambda m: number_words(int(m.group())), text)


# ---- document formats -----------------------------------------------------------------------


def dialogue(turns: Sequence[tuple[str, str]]) -> str:
    """Turns as "Speaker: text" lines; a turn with no speaker is a scene line in parentheses.
    A speaker's consecutive turns are joined into one, and a repeated line is dropped."""
    merged: list[tuple[str, str]] = []
    for speaker, text in turns:
        if merged and merged[-1] == (speaker, text):  # a repeated scene line or message
            continue
        if merged and speaker and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + " " + text)
        else:
            merged.append((speaker, text))
    return "\n".join(f"{s}: {t}" if s else f"({t.rstrip('.')}.)" for s, t in merged)


def _quote(text: str) -> str:
    return f"“{text}”"


def ways_list(about: str, requests: Sequence[str], rng: np.random.Generator) -> str:
    """One request in several wordings, under a header saying what they ask for."""
    header = pick(
        rng,
        (
            f"{about[:1].upper()}{about[1:]}. People say:",
            f"Ways to ask for this: {about}.",
            f"The same request in different words ({about}):",
        ),
    )
    bullet = pick(rng, ("- ", "* ", ""))
    return "\n".join([header, *(bullet + r for r in requests)])


def tool_card(name: str, about: str, requests: Sequence[str], rng: np.random.Generator) -> str:
    """A tool's name and purpose, and requests that call it."""
    if rng.random() < 0.5:
        said = "; ".join(_quote(r) for r in requests)
        return f"Tool: {name}\nWhat it does: {about}\nPeople say: {said}"
    lines = "\n".join(f"- {r}" for r in requests)
    return f"{name}: {about}.\nRequests that use it:\n{lines}"


def scenes(about: str, requests: Sequence[str], ctx: Context) -> str:
    """A short scene per request: someone asks the assistant to do ``about`` (a verb phrase)."""
    lines = []
    for request in requests:
        name = ctx.name()
        line = pick(
            ctx.rng,
            (
                f"{name} wanted to {about}, so {name} said: {_quote(request)}",
                f"{_quote(request)} {name} said to the assistant. {name} wanted to {about}.",
                f"{name} asked the assistant to {about}: {_quote(request)}",
            ),
        )
        lines.append(line)
    return " ".join(lines)


@dataclass(frozen=True)
class Intent:
    """A group of requests that ask for one thing, from an intent dataset."""

    name: str  # a tool-like name, e.g. "alarm_set"
    about: str  # what they ask for, e.g. "set an alarm"
    requests: tuple[str, ...]
    verb: bool = False  # ``about`` is a verb phrase, so scenes can say "wanted to {about}"


CHUNK_LOW, CHUNK_HIGH = 6, 14  # requests per "ways" list or tool card


def intent_docs(intents: Iterable[Intent], ctx: Context) -> Iterator[Doc]:
    """Each intent's kept requests, deduplicated and shuffled, cut into "ways" lists and tool
    cards (and scenes, when ``about`` is a verb phrase)."""
    for intent in intents:
        requests = list(dict.fromkeys(clean(r) for r in intent.requests if clean(r)))
        requests = [r for r in requests if ctx.keep(r)]
        for part in chunks(shuffled(ctx.rng, requests), ctx.rng, CHUNK_LOW, CHUNK_HIGH):
            roll = ctx.rng.random()
            if intent.verb and roll < 0.25:
                yield Doc(scenes(intent.about, part, ctx), "scene")
            elif roll < 0.6:
                yield Doc(ways_list(intent.about, part, ctx.rng), "ways")
            else:
                yield Doc(tool_card(intent.name, intent.about, part, ctx.rng), "card")
