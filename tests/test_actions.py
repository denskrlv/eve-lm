from typing import Any

from eve.data.generate.actions import part_actions, row_actions
from eve.split.words import Span


def _texts(text: str, actions: list[list[Span]]) -> list[str]:
    return [" ".join(text[a:b] for a, b in action) for action in actions]


def test_part_actions_skip_connectors_and_decoration() -> None:
    text = "Please turn off the fan, and then lock the door thanks"
    actions = part_actions(text, ["turn off the fan", "lock the door"])
    assert actions is not None
    assert _texts(text, actions) == ["turn off the fan", "lock the door"]


def test_part_actions_find_a_capitalized_first_part() -> None:
    text = "Turn off the fan and lock the door"
    actions = part_actions(text, ["turn off the fan", "lock the door"])
    assert actions is not None and _texts(text, actions) == ["Turn off the fan", "lock the door"]


def test_part_actions_missing_part() -> None:
    assert part_actions("turn off the fan", ["turn off the fan", "lock the door"]) is None


def _row(source: str, n_calls: int, **extra: Any) -> dict[str, Any]:
    call = {"tool": "set_lights", "args": {"room": "kitchen", "on": False}}
    text = "lights off"
    return {"utterance": text, "source": source, "calls": [call] * n_calls, **extra}


def test_row_actions_rules() -> None:
    assert row_actions(_row("massive", 1)) == [[(0, 10)]]
    assert row_actions(_row("template", 0)) == [[(0, 10)]]
    assert row_actions(_row("xlam", 1)) is None  # xLAM rows are labeled by the converter
    assert row_actions(_row("chat", 2)) is None  # no parts to cut at
    assert row_actions(_row("chat", 2, actions=[[[0, 6]], [[7, 10]]])) == [[(0, 6)], [(7, 10)]]
