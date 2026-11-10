"""Stage B: a model writes one tool's requests, each to its own card of constraints (who says
it, where, in what form), and code checks each one.

A rejected request goes back with its reasons, and the model rewrites only those, up to
``ROUNDS`` times.
"""

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eve.data.datagen.checks import ASK_TYPES, STOPWORDS, row_problems
from eve.data.datagen.llm import Generate
from eve.data.datagen.shard import GenTool, Row
from eve.data.generate.validate import normalize

ROUNDS = 3
CARDS = {"call": 16, "ask": 2, "near": 2}  # requests per tool, by kind
INTENT = {"call": "call", "ask": "ask", "near": "unsupported"}
CARD_KEYS = ("speaker", "situation", "form", "length", "tone")

REQUESTS_SYSTEM = """You write what real people say or type to a voice assistant. You get one \
tool the assistant has and a list of cards. Write one request per card.

Each card says who is speaking, their situation, the form, the length and the tone: write it \
the way that person would say it then. Requests must differ from each other in words and shape.

By the card's intent:
- call: ask for exactly what the tool does. Give "args": a value for every required argument \
and any optional ones you mention. Every text and number value must appear in the request \
exactly as written in args (a number may be written as a word: 7 or seven).
- ask: ask for what the tool does but leave out one required text or number value; put "?" as \
that value in "args".
- near: ask for something close to what the tool does, but that this tool can't do (a \
different action on the same things). "args" is null.

Never use any of the words listed as forbidden, in any form (no "pests" when "pest" is \
forbidden). Say it in other words. Answer with {"rows": [{"card": n, "utterance": "...", \
"args": {...} or null}]}."""

REQUESTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card": {"type": "integer"},
                    "utterance": {"type": "string"},
                    "args": {"type": ["object", "null"]},
                },
                "required": ["card", "utterance", "args"],
            },
        }
    },
    "required": ["rows"],
}


@dataclass
class RowStats:
    """What stage B did, summed over tools."""

    written: int = 0
    rejected: Counter[str] = field(default_factory=Counter)  # by kind
    rounds: Counter[int] = field(default_factory=Counter)  # the round a kept row passed in
    dropped: int = 0  # cards still unwritten after the last round
    tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "RowStats") -> None:
        """Add another tool's stats to these."""
        self.written += other.written
        self.rejected.update(other.rejected)
        self.rounds.update(other.rounds)
        self.dropped += other.dropped
        self.tokens += other.tokens
        self.seconds += other.seconds

    def to_dict(self) -> dict[str, Any]:
        """The JSON form, for stats.json."""
        rounds = {str(k): v for k, v in self.rounds.items()}
        return {**self.__dict__, "rejected": dict(self.rejected), "rounds": rounds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RowStats":
        """Back from stats.json."""
        rounds = Counter({int(k): v for k, v in data.get("rounds", {}).items()})
        return cls(**{**data, "rejected": Counter(data.get("rejected", {})), "rounds": rounds})


def draw_cards(
    tool: GenTool, constraints: Mapping[str, Sequence[str]], rng: np.random.Generator
) -> list[dict[str, str]]:
    """One card per request to write: ``CARDS`` of each kind, ask cards turned into call cards
    when the tool has no required text or number to leave out."""
    counts = dict(CARDS)
    if not any(a.required and a.type in ASK_TYPES for a in tool.args):
        counts["call"] += counts.pop("ask")
    cards = []
    for kind, count in counts.items():
        for _ in range(count):
            card = {"intent": kind}
            card |= {key: str(rng.choice(constraints[key])) for key in CARD_KEYS}
            cards.append(card)
    return cards


def _avoid(tool: GenTool) -> list[str]:
    words = {w for text in tool.texts for w in re.findall(r"[a-z0-9]+", text.lower())}
    return sorted(words - STOPWORDS)


def _kind(problems: Sequence[str]) -> str:
    text = " ".join(problems)
    for needle, kind in (
        ("not in the utterance", "value"),
        ("shares", "shared"),
        ("repeats", "repeat"),
    ):
        if needle in text:
            return kind
    return "schema"


def _prompt(
    tool: GenTool, cards: Mapping[int, dict[str, str]], feedback: Mapping[int, dict[str, Any]]
) -> str:
    lines = [
        "Tool names: " + ", ".join(tool.names),
        "What it does: " + " / ".join(tool.descriptions),
        "Arguments: " + json.dumps([a.to_dict() for a in tool.args], ensure_ascii=False),
        "Never use these words: " + ", ".join(_avoid(tool)),
        "Cards:",
    ]
    for n, card in cards.items():
        line: dict[str, Any] = {"card": n, **card}
        if n in feedback:
            line |= feedback[n]
        lines.append(json.dumps(line, ensure_ascii=False))
    if feedback:
        lines.append(
            'Cards with "rejected" were written before and broke the rules in "because": '
            "write them again."
        )
    return "\n".join(lines)


def write_requests(
    generate: Generate,
    tool: GenTool,
    cards: Sequence[dict[str, str]],
    seen: set[str],
    writer: str,
) -> tuple[list[Row], RowStats]:
    """One request per card for ``tool``, ids ``<tool id>-r01``… by card. ``seen`` holds the
    normalized requests already kept (this shard and earlier ones) and gains the new ones."""
    stats = RowStats()
    pending = {n: card for n, card in enumerate(cards, start=1)}
    kept: dict[int, Row] = {}
    feedback: dict[int, dict[str, Any]] = {}
    for round_number in range(1, ROUNDS + 1):
        if not pending:
            break
        reply = generate(REQUESTS_SYSTEM, _prompt(tool, pending, feedback), REQUESTS_SCHEMA)
        stats.tokens += reply.tokens
        stats.seconds += reply.seconds
        feedback = {}
        items = reply.data.get("rows") if isinstance(reply.data, dict) else None
        answered: set[int] = set()
        for item in items if isinstance(items, list) else []:
            n = item.get("card") if isinstance(item, dict) else None
            if not isinstance(n, int) or n not in pending or n in answered:
                stats.rejected["extra"] += 1
                continue
            answered.add(n)
            stats.written += 1
            card, utterance, args = pending[n], item.get("utterance"), item.get("args")
            intent = INTENT[card["intent"]]
            if intent == "unsupported":
                args = None
            if not isinstance(utterance, str) or (
                intent != "unsupported" and not isinstance(args, dict)
            ):
                stats.rejected["schema"] += 1
                feedback[n] = {"rejected": utterance, "because": ["needs an utterance and args"]}
                continue
            row = Row(f"{tool.id}-r{n:02d}", tool.id, utterance.strip(), intent, args, card, writer)
            if problems := row_problems(row, tool, seen):
                stats.rejected[_kind(problems)] += 1
                feedback[n] = {"rejected": row.utterance, "because": problems}
                continue
            kept[n] = row
            seen.add(normalize(row.utterance))
            stats.rounds[round_number] += 1
            del pending[n]
        for n in pending.keys() - answered:
            stats.rejected["missing"] += 1
            feedback[n] = {"rejected": None, "because": ["not written"]}
    stats.dropped = len(pending)
    return [kept[n] for n in sorted(kept)], stats
