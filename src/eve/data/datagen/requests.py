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
from eve.data.generate.validate import mentions, normalize, value_forms
from eve.data.schema import ASK_VALUE

ROUNDS = 3
CARDS = {"call": 16, "ask": 2, "near": 2}  # requests per tool, by kind
INTENT = {"call": "call", "ask": "ask", "near": "unsupported"}
CARD_KEYS = ("speaker", "situation", "form", "length", "tone")

REQUESTS_SYSTEM = """You write what real people say or type to a voice assistant. You get one \
tool the assistant has and a list of cards. Write one request per card.

Each card says who is speaking, their situation, the form, the length and the tone: write it \
the way that person would naturally say it then. Every request makes clear what the user \
wants the assistant to do (call and ask) or to give them (near): never a remark that asks for \
nothing ("my kale looks sad" alone is not a request). Requests must differ from each \
other in words and shape. Don't pad them with filler like "now", "right now", "please" or \
"I require" unless the card's tone calls for it.

By the card's intent:
- call: ask for exactly what the tool does. Give "args": a value for every required argument \
and any optional one the request mentions, and nothing the request doesn't say. Each text value \
is only the words for that argument, copied exactly from the request ("kale", not "my kale \
right now"). A number may be written as a word in the request (7 or seven), and is said \
in the argument's own unit (minutes=60 is "60 minutes", not "an hour").
- ask: the card names in "leave_out" one argument the request must not say. Request the \
tool's action itself, as in call, just without that value (for a fan timer with leave_out \
"minutes": "turn the fan off later", which wants the action but doesn't say how long; not a \
question like "how long until it stops?"). In "args" still give a value for it, as if the user \
had said it, and every other value as in call.
- near: ask for a different action this tool cannot do at all, about the same kind of things \
(for a tool that waters plants: "repot my fern"). "args" is null.

A call or ask card lists words to "avoid": never use them, in any form ("pests" counts as \
"pest"), and say it in other words. Answer with {"rows": [{"card": n, "utterance": "...", \
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
    log: list[dict[str, Any]] = field(default_factory=list)  # every reject; not in stats.json

    def reject(self, kind: str, **details: Any) -> None:
        """Count a reject of ``kind`` and log it with ``details``."""
        self.rejected[kind] += 1
        self.log.append({**details, "kind": kind})

    def add(self, other: "RowStats") -> None:
        """Add another tool's stats to these."""
        self.log += other.log
        self.written += other.written
        self.rejected.update(other.rejected)
        self.rounds.update(other.rounds)
        self.dropped += other.dropped
        self.tokens += other.tokens
        self.seconds += other.seconds

    def to_dict(self) -> dict[str, Any]:
        """The JSON form, for stats.json."""
        rounds = {str(k): v for k, v in self.rounds.items()}
        data = {k: v for k, v in self.__dict__.items() if k != "log"}
        return {**data, "rejected": dict(self.rejected), "rounds": rounds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RowStats":
        """Back from stats.json."""
        rounds = Counter({int(k): v for k, v in data.get("rounds", {}).items()})
        data = {k: v for k, v in data.items() if k != "log"}
        return cls(**{**data, "rejected": Counter(data.get("rejected", {})), "rounds": rounds})


def draw_cards(
    tool: GenTool, constraints: Mapping[str, Sequence[str]], rng: np.random.Generator
) -> list[dict[str, str]]:
    """One card per request to write: ``CARDS`` of each kind, ask cards turned into call cards
    when the tool has no required text or number to leave out."""
    counts = dict(CARDS)
    askable = [a.name for a in tool.args if a.required and a.type in ASK_TYPES]
    if not askable:
        counts["call"] += counts.pop("ask")
    cards = []
    for kind, count in counts.items():
        for _ in range(count):
            card = {"intent": kind}
            card |= {key: str(rng.choice(constraints[key])) for key in CARD_KEYS}
            if kind == "ask":
                card["leave_out"] = str(rng.choice(askable))
            cards.append(card)
    return cards


def _avoid(tool: GenTool, n: int) -> list[str]:
    """The words card ``n`` must avoid: those of one name and one description, a different
    pair from card to card. Avoiding one pair is enough (see ``checks.clean_pairs``)."""
    pair = (
        tool.names[n % len(tool.names)].replace("_", " "),
        tool.descriptions[n % len(tool.descriptions)],
    )
    words = {w for text in pair for w in re.findall(r"[a-z0-9]+", text.lower())}
    return sorted(words - STOPWORDS)


def _kind(problems: Sequence[str]) -> str:
    text = " ".join(problems)
    for needle, kind in (
        ("not in the utterance", "value"),
        ("the request's case", "value"),
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
        "Cards:",
    ]
    for n, card in cards.items():
        line: dict[str, Any] = {"card": n, **card}
        if card["intent"] != "near":
            line["avoid"] = _avoid(tool, n - 1)
        if n in feedback:
            line |= feedback[n]
        lines.append(json.dumps(line, ensure_ascii=False))
    if feedback:
        lines.append(
            'Cards with "rejected" were written before and broke the rules in "because": '
            "write them again."
        )
    return "\n".join(lines)


def _leave_out(
    tool: GenTool, utterance: str, args: dict[str, Any], name: str | None
) -> tuple[str, list[str], dict[str, Any]]:
    """An ASK card names the argument its request leaves out (``draw_cards``), and the reply
    still gives a value for it: check that the request really doesn't say it, then mark it
    "?". Returns (reject kind, reasons, args)."""
    askable = [a for a in tool.args if a.required and a.type in ASK_TYPES]
    arg = next((a for a in askable if a.name == name), askable[0] if askable else None)
    if arg is None:
        return "schema", ["this tool has no required text or number to leave out"], args
    if arg.name not in args:
        reason = f"give a value for {arg.name} in args, though the request leaves it out"
        return "schema", [reason], args
    if (problem := arg.check(args[arg.name])) is not None:  # before turning it into text
        return "schema", [f"{arg.name}: {problem}"], args
    if any(mentions(utterance, form) for form in value_forms(arg, args[arg.name])):
        reason = f"the request says {arg.name}={args[arg.name]!r}: this card leaves it out"
        return "ask", [reason], args
    return "", [], {**args, arg.name: ASK_VALUE}


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
        where = {"tool": tool.id, "round": round_number}
        for item in items if isinstance(items, list) else []:
            n = item.get("card") if isinstance(item, dict) else None
            if not isinstance(n, int) or n not in pending or n in answered:
                stats.reject("extra", **where, card=n, utterance=None, because=[str(item)[:200]])
                continue
            answered.add(n)
            stats.written += 1
            card, utterance, args = pending[n], item.get("utterance"), item.get("args")
            intent = INTENT[card["intent"]]
            if intent == "unsupported":
                args = None
            else:  # null means no arguments, and a null value means that one isn't given
                args = {} if args is None else args
                if isinstance(args, dict):
                    args = {k: v for k, v in args.items() if v is not None}
            if not isinstance(utterance, str) or (
                intent != "unsupported" and not isinstance(args, dict)
            ):
                because = ["needs an utterance and args"]
                stats.reject("schema", **where, card=n, utterance=utterance, because=because)
                feedback[n] = {"rejected": utterance, "because": because}
                continue
            if intent == "ask" and ASK_VALUE not in args.values():
                kind, because, args = _leave_out(tool, utterance, args, card.get("leave_out"))
                if because:
                    stats.reject(kind, **where, card=n, utterance=utterance, because=because)
                    feedback[n] = {"rejected": utterance, "because": because}
                    continue
            row = Row(f"{tool.id}-r{n:02d}", tool.id, utterance.strip(), intent, args, card, writer)
            if problems := row_problems(row, tool, seen):
                stats.reject(
                    _kind(problems), **where, card=n, utterance=row.utterance, because=problems
                )
                feedback[n] = {"rejected": row.utterance, "because": problems}
                continue
            kept[n] = row
            seen.add(normalize(row.utterance))
            stats.rounds[round_number] += 1
            del pending[n]
        for n in pending.keys() - answered:
            stats.reject("missing", **where, card=n, utterance=None, because=["not written"])
            feedback[n] = {"rejected": None, "because": ["not written"]}
    stats.dropped = len(pending)
    return [kept[n] for n in sorted(kept)], stats
