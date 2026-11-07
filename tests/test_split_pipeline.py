from pathlib import Path
from typing import Any

import pytest
import torch
from helpers import examples, save_joint, write_task

from eve.config import preset
from eve.data.render import Call, TaskExample
from eve.data.schema import Tool
from eve.infer.predict import Answers, Value, WrittenCall, answer_questions
from eve.model.checkpoint import Checkpoint, load_eve
from eve.model.eve import Eve
from eve.model.splitter import Splitter, SplitterConfig
from eve.split import pipeline
from eve.split.decide import Thresholds
from eve.split.pipeline import (
    Predicted,
    combine_intent,
    evaluate_pipeline,
    hard_set_scores,
    outcome_of,
    respond,
    split_metrics,
)
from eve.tokenizer.bpe import BPETokenizer


@pytest.fixture
def eve(tmp_path: Path, tokenizer: BPETokenizer) -> Checkpoint[Eve]:
    return load_eve(save_joint(tmp_path / "eve.pt", tokenizer))


@pytest.fixture
def splitter(tokenizer: BPETokenizer) -> Checkpoint[Splitter]:
    torch.manual_seed(0)
    return Checkpoint(Splitter(SplitterConfig(**preset("tiny").to_dict())), tokenizer, {})


def test_alone_matches_answer_questions(eve: Checkpoint[Eve], library: dict[str, Tool]) -> None:
    ex = examples(library)[1]
    predicted = respond(eve, None, None, ex)
    answers = answer_questions(eve.model, eve.tokenizer, ex.utterance, ex.tools)
    assert predicted.calls == answers.written and predicted.actions == [ex.utterance]


def test_combine_intent() -> None:
    assert combine_intent(["call", "call"]) == "call"
    assert combine_intent(["call", "ask"]) == "ask"
    assert combine_intent(["unsupported", "ask"]) == "unsupported"


def _lights(library: dict[str, Tool], *calls: Call) -> TaskExample:
    return TaskExample("x", (library["set_lights"], library["play_music"]), "call", calls)


def _written(tool: str, **values: Value) -> WrittenCall:
    return WrittenCall(tool, 0.9, values)


def test_outcome_of_counts_values(library: dict[str, Tool]) -> None:
    ex = _lights(library, Call("set_lights", {"room": "hallway", "on": False}))
    said = _written("set_lights", room=Value("value", "hallway"), on=Value("value", True))
    outcome = outcome_of(ex, Predicted("call", [said], ["x"]))
    assert outcome.tools_ok and outcome.values == 2 and outcome.values_right == 1


def test_outcome_of_wrong_order(library: dict[str, Tool]) -> None:
    lights = Call("set_lights", {"room": "hallway", "on": False})
    music = Call("play_music", {"query": "jazz"})
    ex = _lights(library, lights, music)
    said = [_written("play_music"), _written("set_lights")]
    assert not outcome_of(ex, Predicted("call", said, ["x"])).tools_ok


def test_outcome_of_ask(library: dict[str, Tool]) -> None:
    ex = TaskExample(
        "turn on the lights",
        (library["set_lights"],),
        "ask",
        (Call("set_lights", {"room": "?", "on": True}),),
    )
    said = _written("set_lights", room=Value("ask"), on=Value("value", True))
    outcome = outcome_of(ex, Predicted("ask", [said], ["x"]))
    assert outcome.values_right == outcome.values == 2


def test_split_actions_joined_in_order(
    eve: Checkpoint[Eve],
    splitter: Checkpoint[Splitter],
    library: dict[str, Tool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "split_request", lambda *_: ["first", "second"])

    def fake(model: Any, tokenizer: Any, text: str, tools: Any, calibration: Any = None) -> Answers:
        call = WrittenCall(f"tool_{text}", 1.0, {})
        intent = {"call": 1.0, "ask": 0.0, "unsupported": 0.0, "none": 0.0}
        return Answers(intent, [], [call])

    monkeypatch.setattr(pipeline, "answer_questions", fake)
    predicted = respond(eve, splitter, Thresholds(0.5, 0.5), examples(library)[1])
    assert [c.tool for c in predicted.calls] == ["tool_first", "tool_second"]
    assert predicted.actions == ["first", "second"] and predicted.intent == "call"


def test_evaluate_pipeline_groups(
    tmp_path: Path,
    eve: Checkpoint[Eve],
    splitter: Checkpoint[Splitter],
    library: dict[str, Tool],
) -> None:
    write_task(tmp_path / "task", library)
    th = Thresholds(0.5, 0.5)
    results = evaluate_pipeline(eve, splitter, th, tmp_path / "task/val.jsonl", library)
    assert set(results) == {"alone", "split"}
    assert {"all", "0", "1", "2"} <= set(results["alone"])
    assert "exact_acc" in results["split"]["all"]


def test_split_metrics_and_hard_set(
    splitter: Checkpoint[Splitter], monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "turn off the fan and lock the door"
    rows = [
        {
            "utterance": text,
            "source": "template",
            "calls": [{}, {}],
            "actions": [[[0, 16]], [[21, 34]]],
        },
        {"utterance": "lights off", "source": "massive", "calls": [{}]},
    ]
    scores = split_metrics(splitter, Thresholds(0.5, 0.5), rows)
    assert {"exact_split", "wrong_split", "unsure"} <= set(scores)
    assert all(0 <= scores[k] <= 1 for k in ("exact_split", "wrong_split", "unsure"))

    monkeypatch.setattr(
        pipeline, "split_request", lambda *_: ["Turn off the fan.", "lock the door"]
    )
    hard = [
        {"utterance": text, "actions": ["turn off the fan", "lock the door"], "kind": "plain"},
        {"utterance": text, "actions": [text], "kind": "no-split"},
    ]
    assert hard_set_scores(splitter, Thresholds(0.5, 0.5), hard) == {
        "plain": 1.0,
        "no-split": 0.0,
        "all": 0.5,
    }
