"""Builders that several test modules share."""

import json
from pathlib import Path

from eve.data.render import Call, TaskExample
from eve.data.schema import Tool


def examples(library: dict[str, Tool]) -> list[TaskExample]:
    """Six labelled requests, one or more of each intent, over ``library``'s tools."""

    def ex(utterance, tools, intent, *calls):
        return TaskExample(utterance, tuple(library[t] for t in tools), intent, calls)

    return [
        ex(
            "turn off the kitchen lights",
            ["set_lights", "set_thermostat", "play_music"],
            "call",
            Call("set_lights", {"room": "kitchen", "on": False}),
        ),
        ex(
            "dim the bedroom lights to 30% and play some jazz",
            ["play_music", "lock_door", "set_lights"],
            "call",
            Call("set_lights", {"room": "bedroom", "on": True, "brightness": 30}),
            Call("play_music", {"query": "jazz"}),
        ),
        ex(
            "turn on the lights",
            ["set_lights", "set_blinds"],
            "ask",
            Call("set_lights", {"room": "?", "on": True}),
        ),
        ex("can you change it", ["set_thermostat", "set_blinds"], "ask"),  # no slot labels
        ex("order me a pizza", ["set_lights", "play_music", "lock_door"], "unsupported"),
        ex("thanks, that's all", ["set_volume"], "none"),
    ]


def write_task(path: Path, library: dict[str, Tool]) -> None:
    """train.jsonl and the sets fine-tuning reads, all in ``path`` (its task and evals dir)."""
    path.mkdir()
    rows = [e.to_json() for e in examples(library)]
    for name in ("train", "val", "heldout-templates"):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
