"""Splitter labels: which parts of a request are which action.

A row's ``"actions"`` is a list of actions, each a list of ``[start, end]`` character spans of
its ``utterance``. Template rows get them from the sampler's parts (``part_actions``), xLAM
rows from rules (``xlam_actions``). Rows without them are one action when that is certain
(``row_actions``).
"""

from collections.abc import Sequence
from typing import Any

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
