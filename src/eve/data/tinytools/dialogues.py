"""Dialogues (kind D): Taskmaster-1/2/3, STAR and ABCD, plus STAR's APIs as tool cards."""

import ast
import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from eve.data.tinytools.common import Context, Doc, chunks, clean, dialogue, humanize, tool_card

WINDOW_LOW, WINDOW_HIGH = 4, 8  # Taskmaster turns per document
MIN_TURNS = 3  # a shorter window or dialogue is left out
TASKMASTER_SPEAKERS = {"USER": "User", "ASSISTANT": "Assistant"}
TASKMASTER_TRAVEL = ("flights", "hotels")  # Taskmaster-2 files left out
STAR_TRAVEL = ("hotel", "plane")  # STAR domains left out
STAR_OPENER_WORDS = 4  # a user's first turn shorter than this ("Hello") is no request


def _windows(turns: list[tuple[str, str]], ctx: Context) -> Iterator[list[tuple[str, str]]]:
    for window in chunks(turns, ctx.rng, WINDOW_LOW, WINDOW_HIGH):
        if len(window) >= MIN_TURNS:
            yield window


def taskmaster(paths: list[Path], ctx: Context) -> Iterator[Doc]:
    """Taskmaster dialogues cut into windows of 4-8 turns; a window with a travel topic or an
    eval-set request is left out."""
    for path in paths:
        if path.stem in TASKMASTER_TRAVEL:
            ctx.drop("travel file")
            continue
        for convo in json.loads(path.read_text()):
            turns = [
                (
                    TASKMASTER_SPEAKERS.get(u.get("speaker", "").upper(), "User"),
                    clean(u.get("text")),
                )
                for u in convo.get("utterances") or []
            ]
            turns = [(s, t) for s, t in turns if t]
            for window in _windows(turns, ctx):
                if ctx.keep(t for _, t in window):
                    yield Doc(dialogue(window), "dialogue")


def taskmaster_1(raw: Path, ctx: Context) -> Iterator[Doc]:
    return taskmaster(sorted((raw / "taskmaster/TM-1-2019").glob("*-dialogs.json")), ctx)


def taskmaster_2(raw: Path, ctx: Context) -> Iterator[Doc]:
    return taskmaster(sorted((raw / "taskmaster/TM-2-2020/data").glob("*.json")), ctx)


def taskmaster_3(raw: Path, ctx: Context) -> Iterator[Doc]:
    return taskmaster(sorted((raw / "taskmaster/TM-3-2020/data").glob("data_*.json")), ctx)


# ---- STAR ------------------------------------------------------------------------------------

_CALL = re.compile(r"api\.\w+\((.*)\)")
MAX_VALUE_WORDS = 6  # longer values (a whole fraud report) are left out of the scene line


def _plain(value: str) -> str | None:
    """A constraint's value without its quotes or ``api.is_equal_to(...)``: a quoted text or a
    number, else None."""
    value = value.strip()
    if match := _CALL.fullmatch(value):
        value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = clean(value[1:-1])
    elif not re.fullmatch(r"-?\d+(\.\d+)?", value):
        return None
    return value if value and len(value.split()) <= MAX_VALUE_WORDS else None


def star_constraints(raw: str | list[Any]) -> list[tuple[str, str]]:
    """A query's constraints as (field, value) pairs: [{'Name': 'api.is_equal_to("Dr. Johnson")'}]
    -> [("name", "Dr. Johnson")]. Constraints without a plain value are left out. Most
    dialogues store them as a string, some as the list itself."""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    pairs = []
    for item in raw if isinstance(raw, list) else []:
        for key, value in item.items() if isinstance(item, dict) else []:
            if (plain := _plain(str(value))) is not None:
                pairs.append((humanize(key), plain))
    return pairs


def _star_query(event: dict[str, Any]) -> str:
    """A query as a scene line: the assistant looks something up, with what it knows."""
    what = humanize(event.get("APIName", "the database"))
    pairs = star_constraints(event.get("Constraints", "[]"))
    given = ", ".join(f"{k} {v}" for k, v in pairs)
    return f"The assistant searches {what}" + (f" for {given}" if given else "")


def _star_turns(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The turns people saw: the user's, the wizard's sent messages and queries (as scene lines).
    Suggestions the wizard didn't send and the guides' instructions are left out."""
    turns = []
    for event in events:
        agent, action, text = event.get("Agent"), event.get("Action"), clean(event.get("Text", ""))
        if agent == "User" and action == "utter" and text:
            turns.append(("User", text))
        elif agent == "Wizard" and action in ("utter", "pick_suggestion") and text:
            turns.append(("Assistant", text))
        elif agent == "Wizard" and action == "query":
            turns.append(("", _star_query(event)))
    return turns


def _star_dialogues(raw: Path) -> Iterator[dict[str, Any]]:
    for path in sorted((raw / "star/dialogues").glob("*.json"), key=lambda p: int(p.stem)):
        yield json.loads(path.read_text())


def _star_tasks(convo: dict[str, Any]) -> list[str]:
    return [c["Task"] for c in convo["Scenario"].get("WizardCapabilities", []) if c.get("Task")]


def _star_travel(convo: dict[str, Any]) -> bool:
    domains = [d for d in convo["Scenario"].get("Domains") or [] if d] + _star_tasks(convo)
    return any(d.startswith(STAR_TRAVEL) for d in domains)


def star(raw: Path, ctx: Context) -> Iterator[Doc]:
    """STAR's dialogues, the unhappy ones included, then a tool card per API."""
    openers: dict[str, list[str]] = {}
    for convo in _star_dialogues(raw):
        if _star_travel(convo):
            ctx.drop("travel domain")
            continue
        turns = _star_turns(convo["Events"])
        said = [t for s, t in turns if s]
        if len(said) < MIN_TURNS:
            ctx.drop("too short")
            continue
        if not ctx.keep(said):
            continue
        yield Doc(dialogue(turns), "dialogue")
        tasks = _star_tasks(convo)
        first = next(
            (t for s, t in turns if s == "User" and len(t.split()) >= STAR_OPENER_WORDS), ""
        )
        if len(tasks) == 1 and first:
            openers.setdefault(tasks[0], []).append(first)
    yield from _star_cards(raw, openers, ctx)


def _star_input(field: dict[str, Any]) -> str:
    """An API input in words: "City (one of Chicago, Detroit, ...)"."""
    name = field.get("ReadableName") or humanize(field["Name"])
    if categories := field.get("Categories"):
        shown = ", ".join(categories[:5]) + (", ..." if len(categories) > 5 else "")
        return f"{name} (one of {shown})"
    if "Min" in field and "Max" in field:
        return f"{name} (a number from {field['Min']} to {field['Max']})"
    return str(name)


def _star_cards(raw: Path, openers: dict[str, list[str]], ctx: Context) -> Iterator[Doc]:
    """A card per API: what it takes, and how users opened the dialogues that used it."""
    for task, requests in sorted(openers.items()):
        path = raw / "star/apis" / f"{task}.json"
        if not path.exists():
            continue
        api = json.loads(path.read_text())
        required = set(api.get("required", []))
        inputs = [
            _star_input(f) + (", required" if f["Name"] in required else "")
            for f in api.get("input", [])
        ]
        about = f"{humanize(task)}; it takes {'; '.join(inputs)}" if inputs else humanize(task)
        for part in chunks(requests, ctx.rng, 5, 10):
            yield Doc(tool_card(task, about, part, ctx.rng), "card")


# ---- ABCD ------------------------------------------------------------------------------------

ABCD_SPEAKERS = {"agent": "Agent", "customer": "Customer", "action": ""}


def abcd(raw: Path, ctx: Context) -> Iterator[Doc]:
    """ABCD's customer service chats, with the agent's actions as scene lines."""
    with gzip.open(raw / "abcd/abcd_v1.1.json.gz", "rt") as f:
        splits = json.load(f)
    for split in ("train", "dev", "test"):
        for convo in splits.get(split, []):
            turns = [
                (ABCD_SPEAKERS.get(speaker, "Customer"), clean(text))
                for speaker, text in convo["original"]
                if clean(text)
            ]
            said = [t for s, t in turns if s]
            if len(said) < MIN_TURNS:
                ctx.drop("too short")
            elif ctx.keep(said):
                yield Doc(dialogue(turns), "dialogue")
