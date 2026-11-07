"""Splitter labels: which parts of a request are which action.

A row's ``"actions"`` is a list of actions, each a list of ``[start, end]`` character spans of
its ``utterance``. Template rows get them from the sampler's parts (``part_actions``), xLAM
rows from rules (``xlam_actions``). Rows without them are one action when that is certain
(``row_actions``).
"""

import re
from collections.abc import Sequence
from typing import Any

from eve.data.generate.validate import value_forms
from eve.data.render import Call, TaskExample
from eve.split.words import Span

Actions = list[list[Span]]


def part_actions(utterance: str, parts: Sequence[str]) -> Actions | None:
    """One action per part, each found in order in ``utterance`` (which may add a prefix, a
    suffix and a capital first letter); None if a part isn't there."""
    lower = utterance.lower()
    if len(lower) != len(utterance):  # lower() changed the length: offsets would be off
        lower = utterance
    actions, start = [], 0
    for part in parts:
        at = lower.find(part.lower(), start)
        if at < 0:
            return None
        actions.append([(at, at + len(part))])
        start = at + len(part)
    return actions


def row_actions(row: dict[str, Any]) -> Actions | None:
    """A task row's actions: its ``"actions"`` if it has them; else, unless it is from xLAM
    (whose converter labels every request it can), one action for the whole request if it has
    at most one call; else None (unknown)."""
    if "actions" in row:
        return [[(int(a), int(b)) for a, b in action] for action in row["actions"]]
    if row.get("source") == "xlam" or len(row.get("calls", ())) > 1:
        return None
    return [[(0, len(row["utterance"]))]]


# ---- xLAM -----------------------------------------------------------------------------------
# Labels from the gold calls (md/2026-11-07-splitter-plan.md, Task 5): each call is anchored to
# its argument values in the request; between two calls' anchors there must be exactly one
# connector, which is where the request is cut; an action that starts without its own head
# borrows the previous action's. Anything unclear is dropped (None), not guessed.

CONNECTORS = (
    "and", "also", "additionally", "then", "furthermore", "moreover", "finally", "lastly",
    "after that", ".", "?", "!", ";",
)  # fmt: skip
_CONNECTOR = re.compile(
    r"(?<!\w)(?:and|also|additionally|then|furthermore|moreover|finally|lastly|after that)(?!\w)"
    r"|[.?!;](?=\s|$)",
    re.I,
)
_JOINS = re.compile(r"[\s,]*")  # what may sit between two connectors that count as one
_DETERMINER = re.compile(r"(?<!\w)(?:the|a|an|my|their)(?!\w)", re.I)
_HEADLESS = re.compile(r"(?:the|a|an|my|their)(?!\w)|\d", re.I)
_PREPOSITION = r"(?:in|for|of|at|on|from|to|about)"
_BARE = re.compile(rf"(?:{_PREPOSITION}\s+)?(['\"]?)", re.I)  # all that may come before a value
_DEPENDENT = re.compile(rf"(?:{_PREPOSITION}|if|whether|with|by)(?!\w)", re.I)  # needs a head
_QUOTES = "'\""
_LEAD, _TRAIL = " \t\n,", " \t\n,;.?!"


def _find(text: str, form: str, taken: list[Span]) -> Span | None:
    """The first whole-word, any-case match of ``form`` that overlaps no ``taken`` span."""
    for match in re.finditer(rf"(?<!\w){re.escape(form)}(?!\w)", text, re.IGNORECASE):
        span = match.span()
        if not any(span[0] < b and a < span[1] for a, b in taken):
            return span
    return None


def _anchor(example: TaskExample, call: Call, taken: list[Span]) -> list[Span] | None:
    """Where the call's str and number values are in the request ([] if it has none); None if
    one isn't there. A str value like "WMT,TGT" is found piece by piece."""
    tool = next(t for t in example.tools if t.name == call.tool)
    spans: list[Span] = []
    for name, value in call.args.items():
        arg = tool.arg(name)
        if arg.type not in ("str", "int", "float"):
            continue
        pieces = [p.strip() for p in str(value).split(",")] if arg.type == "str" else [value]
        for piece in pieces:
            forms = value_forms(arg, piece) if arg.type != "str" else [str(piece)]
            span = next((s for f in forms if f and (s := _find(example.utterance, f, taken))), None)
            if span is None:
                return None
            taken.append(span)
            spans.append(span)
    return spans


def _connectors(text: str, start: int, end: int) -> list[Span]:
    """The connectors in ``text[start:end]``, adjacent ones ("? Also", ", and then") merged
    into one; a sentence break with only punctuation after it, at the end, isn't one."""
    groups: list[Span] = []
    for match in _CONNECTOR.finditer(text, start, end):
        a, b = match.span()
        if groups and _JOINS.fullmatch(text, groups[-1][1], a):
            groups[-1] = (groups[-1][0], b)
        else:
            groups.append((a, b))
    if end == len(text):
        groups = [g for g in groups if re.search(r"\w", text[g[1] :])]
    return groups


def _trim(text: str, a: int, b: int, trail: str = _TRAIL) -> Span:
    while a < b and text[a] in _LEAD:
        a += 1
    while b > a and text[b - 1] in trail:
        b -= 1
    return a, b


def _head(text: str, segment: Span, region: Span | None, starts_bare: bool) -> Span | None:
    """The opening of an action that a following headless action borrows. For an action that
    is only a value ("and Bergen"): the words before this one's value ("What is the weather
    in"). Otherwise the words before this one's first determiner ("Fetch" of "Fetch the cat
    breeds"), or, with no determiner before its value, its first word ("Find" of "Find all
    songs by The Weeknd")."""
    a, b = segment
    if starts_bare:
        if region is None:
            return None
        b = region[0]
    elif match := _DETERMINER.search(text, a, region[0] if region else b):
        b = match.start()
    elif first := re.compile(r"\S+").match(text, a):
        b = first.end()
    head = _trim(text, a, b, _TRAIL + _QUOTES)
    return head if head[1] > head[0] else None


def _bare_start(text: str, segment: Span, region: Span | None) -> int | None:
    """Where an action that is only a value starts, its opening quote kept and a preposition
    before it dropped ("in Bergen" -> "Bergen"); None if it is more than a value. A number
    isn't a bare value: "10 WikiHow steps" is a count of something."""
    if region is None or text[region[0]].isdigit():
        return None
    match = _BARE.fullmatch(text, segment[0], region[0])
    return None if match is None else match.start(1)


def xlam_actions(example: TaskExample) -> Actions | None:
    """The actions of an xLAM request, from its gold calls; None if the rules can't tell."""
    text = example.utterance
    if len(example.calls) <= 1:
        return [[(0, len(text))]]
    taken: list[Span] = []
    regions: list[Span | None] = []
    for call in example.calls:
        spans = _anchor(example, call, taken)
        if spans is None:
            return None
        regions.append((min(s[0] for s in spans), max(s[1] for s in spans)) if spans else None)
    anchored = [r for r in regions if r is not None]
    if any(r[1] > s[0] for r, s in zip(anchored, anchored[1:], strict=False)):
        return None  # the calls aren't in the request's order

    # Each boundary between two calls lies in the gap between the anchors around it.
    gaps: dict[Span, int] = {}
    for k in range(len(regions) - 1):
        before = next((r[1] for r in reversed(regions[: k + 1]) if r), 0)
        after = next((r[0] for r in regions[k + 1 :] if r), len(text))
        gaps[before, after] = gaps.get((before, after), 0) + 1
    cuts: list[Span] = []
    for (start, end), needed in gaps.items():
        found = _connectors(text, start, end)
        if len(found) != needed:
            return None
        cuts += found
    cuts.sort()

    edges = [0, *(c for cut in cuts for c in cut), len(text)]
    segments = [_trim(text, edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    if any(a >= b for a, b in segments):
        return None
    actions: Actions = [[segments[0]]]
    head: Span | None = None
    for k in range(1, len(segments)):
        a, b = segments[k]
        region, opening = regions[k], text[a:b].lstrip(_QUOTES)
        bare = _bare_start(text, segments[k], region)
        after_sentence = re.search(r"[.?!;]", text[cuts[k - 1][0] : cuts[k - 1][1]]) is not None
        if bare is None and _DEPENDENT.match(opening):
            return None  # "in German for Germany", "if the domain is disposable": no head
        if region is None and opening[:1].isupper() and not after_sentence:
            return None  # "and New Zealand": a name that isn't anchored to its call
        if bare is not None or _HEADLESS.match(opening):
            head = head or _head(text, segments[k - 1], regions[k - 1], bare is not None)
            start = bare if bare is not None and head else a
            actions.append([head, (start, b)] if head else [(a, b)])
        else:
            head = None
            actions.append([(a, b)])
    return actions
