"""Claude's review of a finished shard, and the shard's report.

Claude reads every tool and request and may fix a request or its label, drop a request, or drop a
broken tool with all its requests. It never edits a tool: the other requests depend on it. A
fixed request must pass the code checks again, or it is dropped. A shard is approved when the
fixes and drops stay under ``APPROVE_FIX_RATE`` of the requests reviewed.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from eve.data.datagen.checks import row_problems
from eve.data.datagen.llm import Generate, LLMError
from eve.data.datagen.requests import RowStats
from eve.data.datagen.shard import GenTool, Row
from eve.data.datagen.tools import ToolStats
from eve.data.generate.validate import normalize
from eve.data.schema import Tool

APPROVE_FIX_RATE = 0.10
INTENTS = ("call", "ask", "unsupported")

REVIEW_SYSTEM = """You review training data for a small model that reads a user's request and \
a list of tools and decides what to call. Each tool has several names and descriptions; each \
request was written by another model for one tool, with a label: intent "call" (the tool does \
it, with these args), "ask" (the tool does it but a required value is missing, written "?"), or \
"unsupported" (close to the tool, but the tool can't do it).

The rules every request follows (code has checked them already; keep them in any fix):
- Each text value is only the words for that argument, copied exactly from the request. A \
number value may be written as a word in the request (7 or seven): that is fine, don't change it.
- "?" marks a missing required value, never on an optional argument.
- A call or ask request shares no word (in any form) with at least one of the tool's names and \
one of its descriptions. An unsupported request may use the tool's words.

Check every request:
- The label is right: a call or ask request really asks for what the tool does; an unsupported \
one really can't be done by the tool (if the tool could do it, it is wrong).
- Each value holds exactly the user's words for its argument: nothing missing, invented, or \
extra (for "find the grip errors right now", "grip errors", not "grip errors right now").
- It is a request a real person would say to an assistant, not a remark that asks for nothing.
- It avoids the tool's words honestly: misspelling a tool word to dodge the rule is not allowed.

Check every tool: it must make sense, and it must not do what any of the reserved tools do.

Then answer:
- "fixes": requests you can repair. Give the id and only what changes: "utterance", "intent", \
"args" (the whole args object). Never rename a tool or change its arguments.
- "drop_rows": requests that can't be repaired.
- "drop_tools": tools that are broken or do what a reserved tool does; all their requests go.
Give a few words of "why" for each. Leave good requests alone."""

_WHY = {"why": {"type": "string"}}
REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "utterance": {"type": "string"},
                    "intent": {"type": "string", "enum": list(INTENTS)},
                    "args": {"type": ["object", "null"]},
                    **_WHY,
                },
                "required": ["id", "why"],
            },
        },
        "drop_rows": {
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "string"}, **_WHY}},
        },
        "drop_tools": {
            "type": "array",
            "items": {"type": "object", "properties": {"id": {"type": "string"}, **_WHY}},
        },
    },
    "required": ["fixes", "drop_rows", "drop_tools"],
}


@dataclass
class ReviewResult:
    """A shard after the review."""

    tools: list[GenTool]
    rows: list[Row]
    reviewed: int = 0  # requests not under a dropped tool
    fixed: int = 0
    dropped: int = 0  # dropped by Claude, or fixed and then failing the checks
    tools_dropped: list[str] = field(default_factory=list)
    reasons: list[dict[str, str]] = field(default_factory=list)
    cost: float = 0.0

    @property
    def fix_rate(self) -> float:
        """The share of reviewed requests Claude had to fix or drop."""
        return (self.fixed + self.dropped) / self.reviewed if self.reviewed else 0.0


def _items(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = data.get(key) or []
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def apply_review(data: Any, tools: Sequence[GenTool], rows: Sequence[Row]) -> ReviewResult:
    """Apply Claude's answer (see ``REVIEW_SCHEMA``) to the shard. Raises ``LLMError`` if it
    isn't an answer at all."""
    if not isinstance(data, dict):
        raise LLMError(f"the review answered no JSON object: {str(data)[:200]}")
    reasons: list[dict[str, str]] = []
    known = {t.id for t in tools}
    gone = set()
    for item in _items(data, "drop_tools"):
        tool_id = str(item.get("id"))
        if tool_id in known:
            gone.add(tool_id)
            reasons.append({"id": tool_id, "action": "drop tool", "why": str(item.get("why", ""))})
    by_tool = {t.id: t for t in tools if t.id not in gone}
    live = {r.id: r for r in rows if r.tool in by_tool}
    result = ReviewResult([t for t in tools if t.id not in gone], [], reviewed=len(live))
    result.tools_dropped = sorted(gone)
    rows_of_gone = {r.id for r in rows if r.tool in gone}

    for item in _items(data, "drop_rows"):
        row_id = str(item.get("id"))
        if row_id in live:
            del live[row_id]
            result.dropped += 1
            reasons.append({"id": row_id, "action": "drop", "why": str(item.get("why", ""))})
        elif row_id not in rows_of_gone:
            reasons.append({"id": row_id, "action": "ignored", "why": "unknown id"})

    for item in _items(data, "fixes"):
        row_id = str(item.get("id"))
        if row_id not in live:
            if row_id not in rows_of_gone:
                reasons.append({"id": row_id, "action": "ignored", "why": "unknown id"})
            continue
        old = live[row_id]
        intent = item.get("intent", old.intent)
        args = item.get("args", old.args)
        utterance = item.get("utterance", old.utterance)
        if intent == "unsupported":
            args = None
        if (str(utterance).strip(), intent, args) == (old.utterance, old.intent, old.args):
            reasons.append({"id": row_id, "action": "unchanged", "why": str(item.get("why", ""))})
            continue
        problems = ["the fix isn't a request with a known intent and args"]
        if (
            intent in INTENTS
            and isinstance(utterance, str)
            and (args is None or isinstance(args, dict))
        ):
            new = replace(
                old, utterance=utterance.strip(), intent=intent, args=args, writer="fixed"
            )
            others = {normalize(r.utterance) for r in live.values() if r.id != row_id}
            problems = row_problems(new, by_tool[old.tool], others)
        if problems:
            del live[row_id]
            result.dropped += 1
            reasons.append(
                {"id": row_id, "action": "fix failed checks", "why": "; ".join(problems)}
            )
            continue
        live[row_id] = new
        result.fixed += 1
        reasons.append({"id": row_id, "action": "fix", "why": str(item.get("why", ""))})

    result.rows = [live[r.id] for r in rows if r.id in live]
    result.reasons = reasons
    return result


def review_message(tools: Sequence[GenTool], rows: Sequence[Row], blocked: Sequence[Tool]) -> str:
    """What Claude reads: the reserved tools, then the shard's tools and requests as JSON lines."""
    names = {t.id: t.names[0] for t in tools}
    lines = ["Reserved tools (no tool may do what these do):"]
    lines += [f"- {t.name}: {t.description}" for t in blocked]
    lines.append("Tools:")
    lines += [json.dumps(t.to_dict(), ensure_ascii=False) for t in tools]
    lines.append("Requests:")
    for row in rows:
        line = {
            k: v
            for k, v in row.to_dict(names.get(row.tool, "")).items()
            if k in ("id", "tool", "utterance", "intent", "calls")
        }
        lines.append(json.dumps(line, ensure_ascii=False))
    return "\n".join(lines)


def review_shard(
    generate: Generate,
    tools: Sequence[GenTool],
    rows: Sequence[Row],
    blocked: Sequence[Tool] = (),
) -> ReviewResult:
    """Have Claude review the shard and apply its answer."""
    reply = generate(REVIEW_SYSTEM, review_message(tools, rows, blocked), REVIEW_SCHEMA)
    result = apply_review(reply.data, tools, rows)
    result.cost = reply.cost
    return result


def _per_second(tokens: int, seconds: float) -> float:
    return round(tokens / seconds, 1) if seconds else 0.0


def shard_report(
    tool_stats: ToolStats, row_stats: RowStats, result: ReviewResult
) -> dict[str, Any]:
    """Every count of the shard, and whether it is approved."""
    return {
        "approved": result.fix_rate < APPROVE_FIX_RATE,
        "tools": {**tool_stats.to_dict(), "kept": len(result.tools)},
        "rows": {**row_stats.to_dict(), "kept": len(result.rows)},
        "review": {
            "reviewed": result.reviewed,
            "fixed": result.fixed,
            "dropped": result.dropped,
            "fix_rate": round(result.fix_rate, 4),
            "tools_dropped": result.tools_dropped,
            "cost": result.cost,
            "reasons": result.reasons,
        },
        "speed": {
            "tools_tokens_per_s": _per_second(tool_stats.tokens, tool_stats.seconds),
            "requests_tokens_per_s": _per_second(row_stats.tokens, row_stats.seconds),
        },
    }
