"""Stage A: a model invents the tools of a shard, a few per topic, and code judges each one.

A rejected tool goes back with its reasons, and the model writes a replacement, up to ``ROUNDS``
times per topic.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from eve.data.datagen.checks import tool_problems
from eve.data.datagen.llm import Generate
from eve.data.datagen.meaning import (
    BLOCK_SIMILARITY,
    COHERENCE_FLOOR,
    REPEAT_SIMILARITY,
    TWIN_SIMILARITY,
    MeaningIndex,
)
from eve.data.datagen.shard import GenTool
from eve.data.schema import ARG_TYPES, MAX_ARGS, SchemaError

ROUNDS = 3

TOOLS_SYSTEM = f"""You invent tools for a voice assistant on a phone or in a home: small actions \
or lookups it could really perform for an ordinary person. You get a topic and how many tools \
to write.

For each tool give:
- "names": 2 to 4 snake_case names (lowercase words joined by _), each saying what the tool \
does in different words. The first is the main one.
- "descriptions": 2 to 4 descriptions of what it does, each at most 10 words and in \
different words from the others (not the same sentence with one word swapped).
- "args": what it needs, up to {MAX_ARGS}. Each has "name" (snake_case), "type" (one of \
{", ".join(ARG_TYPES)}), and optionally "required" (default true), "options" (enum only, 2 \
or more), "min" and "max" (int and float only), "description".

Rules:
- One clear, everyday capability per tool, the kind of thing a person really asks a phone or \
smart speaker for, different from every tool listed as taken. Keep it simple: no gimmicks.
- Every name and description of a tool describes the same single action.
- Nothing about flights, hotels, trains, travel bookings, or money currencies.
- Plain everyday English."""

_ARG_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string", "enum": list(ARG_TYPES)},
        "required": {"type": "boolean"},
        "options": {"type": "array", "items": {"type": "string"}},
        "min": {"type": "number"},
        "max": {"type": "number"},
        "description": {"type": "string"},
    },
    "required": ["name", "type"],
}
TOOLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                    "descriptions": {"type": "array", "items": {"type": "string"}},
                    "args": {"type": "array", "items": _ARG_SCHEMA},
                },
                "required": ["names", "descriptions", "args"],
            },
        }
    },
    "required": ["tools"],
}


@dataclass
class ToolStats:
    """What stage A did for one shard."""

    proposed: int = 0
    kept: int = 0
    rejected: Counter[str] = field(default_factory=Counter)  # by kind
    dropped: int = 0  # tools still missing after the last round
    twins: int = 0  # kept with a training tool's capability
    blocked_topics: int = 0  # topics skipped for being too close to an eval tool
    tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """The JSON form, for stats.json."""
        return {**self.__dict__, "rejected": dict(self.rejected)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolStats":
        """Back from stats.json."""
        return cls(**{**data, "rejected": Counter(data.get("rejected", {}))})


class Judge:
    """Decides whether an invented tool is kept: valid, not an eval tool's capability, not a
    repeat of an accepted tool, and (if it twins a training tool) under that tool's capability."""

    def __init__(
        self,
        blocked: MeaningIndex,
        accepted: MeaningIndex,
        library: MeaningIndex,
        capabilities: Mapping[str, str],  # training tool name -> its capability
        taken: set[str],  # every name in use: library, evals, earlier shards
    ):
        self.blocked, self.accepted, self.library = blocked, accepted, library
        self.capabilities, self.taken = capabilities, taken

    def judge(self, tool: GenTool) -> tuple[GenTool | None, str, list[str]]:
        """``(kept tool, "", [])``, ``(kept tool, "twin", [])``, or ``(None, kind, reasons)``."""
        if problems := tool_problems(tool, self.taken):
            kind = "name" if all("taken" in p for p in problems) else "wording"
            return None, kind, problems
        texts = list(tool.descriptions)
        if self.blocked.spread(texts) < COHERENCE_FLOOR:
            return (
                None,
                "incoherent",
                ["the descriptions say different things: describe one action"],
            )
        near = self.blocked.nearest(texts)
        if near and near[1] > BLOCK_SIMILARITY:
            return None, "blocked", [f"too close to the reserved tool {near[0]}: invent another"]
        near = self.accepted.nearest(texts)
        if near and near[1] > REPEAT_SIMILARITY:
            return None, "repeat", [f"too close to {near[0]}, which exists: invent something else"]
        kind = ""
        near = self.library.nearest(texts)
        if near and near[1] > TWIN_SIMILARITY:
            tool = replace(tool, capability=self.capabilities.get(near[0], near[0]))
            kind = "twin"
        self.accepted.add(tool.names[0], texts)
        self.taken.update(tool.names)
        return tool, kind, []


def _clean_arg(arg: Any) -> Any:
    """Drop the empty fields models like to fill in (null bounds, empty option lists)."""
    if not isinstance(arg, dict):
        return arg
    return {k: v for k, v in arg.items() if v is not None and v != [] and v != ""}


def _strings(item: dict[str, Any], key: str) -> list[str]:
    value = item.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SchemaError(f"{key} must be a list of strings, got {value!r}")
    return value


def _parse(item: Any, topic: str) -> GenTool:
    if not isinstance(item, dict):
        raise SchemaError(f"a tool must be an object, got {item!r}")
    args = item.get("args") or []
    if not isinstance(args, list):
        raise SchemaError("args must be a list")
    return GenTool.from_dict(
        {
            "id": "new",
            "topic": topic,
            "names": _strings(item, "names"),
            "descriptions": _strings(item, "descriptions"),
            "args": [_clean_arg(a) for a in args],
        }
    )


def _prompt(
    topic: str,
    wanted: int,
    taken: Sequence[str],
    feedback: Sequence[dict[str, Any]],
) -> str:
    lines = [f"Topic: {topic}", f"Write {wanted} tool{'s' if wanted > 1 else ''}."]
    if taken:
        lines.append("Taken (don't repeat these): " + ", ".join(taken))
    if feedback:
        lines.append(
            f"Asked for {wanted} replacement(s): these were rejected, write new tools instead:"
        )
        lines += [json.dumps(f, ensure_ascii=False) for f in feedback]
    return "\n".join(lines)


def invent_tools(
    generate: Generate,
    topics: Sequence[str],
    judge: Judge,
    per_topic: int,
    topic_done: Callable[[ToolStats], None] = lambda stats: None,
) -> tuple[list[GenTool], ToolStats]:
    """``per_topic`` tools for each topic, ids ``t01``, ``t02``… in order of acceptance.
    ``topic_done`` is called with the stats so far after each topic."""
    stats = ToolStats()
    kept: list[GenTool] = []
    for topic in topics:
        wanted = per_topic
        feedback: list[dict[str, Any]] = []
        for _ in range(ROUNDS):
            if wanted == 0:
                break
            prompt = _prompt(topic, wanted, [t.names[0] for t in kept], feedback)
            reply = generate(TOOLS_SYSTEM, prompt, TOOLS_SCHEMA)
            stats.tokens += reply.tokens
            stats.seconds += reply.seconds
            feedback = []
            items = reply.data.get("tools") if isinstance(reply.data, dict) else None
            if not isinstance(items, list):
                stats.rejected["json"] += 1
                continue
            for item in items[:wanted]:
                stats.proposed += 1
                try:
                    tool = _parse(item, topic)
                except SchemaError as error:
                    stats.rejected["schema"] += 1
                    feedback.append({"rejected": item, "because": [str(error)]})
                    continue
                accepted, kind, reasons = judge.judge(tool)
                if accepted is None:
                    stats.rejected[kind] += 1
                    feedback.append({"rejected": tool.names[0], "because": reasons})
                    continue
                stats.twins += kind == "twin"
                kept.append(replace(accepted, id=f"t{len(kept) + 1:02d}"))
                stats.kept += 1
                wanted -= 1
        stats.dropped += wanted
        topic_done(stats)
    return kept, stats
