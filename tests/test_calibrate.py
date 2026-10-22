import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from eve.calibrate import (
    GRID,
    Reliability,
    TaskSet,
    _answered,
    fit,
    known_tools,
    reliability,
    save_calibration,
)
from eve.cli import main
from eve.config import INTENTS, preset
from eve.data.render import IGNORE, Call, TaskCollator, TaskExample, render
from eve.data.schema import Tool
from eve.infer.predict import Calibration, Calibrations, answer_questions
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve, Output, Questions
from eve.tasks.questions import value_hits
from eve.tokenizer.bpe import BPETokenizer

CFG = preset("tiny")


def examples(library: dict[str, Tool]) -> list[TaskExample]:
    def ex(utterance, tools, intent, *calls):
        return TaskExample(utterance, tuple(library[t] for t in tools), intent, calls)

    return [
        ex(
            "turn off the kitchen lights",
            ["set_lights", "set_thermostat"],
            "call",
            Call("set_lights", {"room": "kitchen", "on": False}),
        ),
        ex(
            "dim the bedroom lights to 30% and play some jazz",
            ["play_music", "set_lights"],
            "call",
            Call("set_lights", {"room": "bedroom", "on": True, "brightness": 30}),
            Call("play_music", {"query": "jazz"}),
        ),
        ex(
            "turn on the lights",
            ["set_lights"],
            "ask",
            Call("set_lights", {"room": "?", "on": True}),
        ),
        ex("can you change it", ["set_thermostat", "set_blinds"], "ask"),  # no slot labels
        ex("order me a pizza", ["set_lights", "play_music"], "unsupported"),
        ex("thanks, that's all", ["set_volume"], "none"),
    ]


@pytest.fixture
def batch(library, tokenizer):
    return TaskCollator(CFG)([render(e, tokenizer, CFG) for e in examples(library)])


class Sure(Eve):
    """Answers every question with logits ``scale`` for one answer and 0 for the others: the
    gold answer, or a wrong one for the examples in ``wrong``."""

    def __init__(self, scale: float, wrong: tuple[int, ...] = ()):
        super().__init__(CFG)
        self.scale, self.wrong = scale, wrong

    def forward(
        self, enc_ids, enc_mask, dec_ids=None, dec_mask=None, tool_pos=None, tool_mask=None
    ):
        batch_size = enc_ids.shape[0]
        flip = torch.zeros(batch_size, dtype=torch.long)
        flip[list(self.wrong)] = 1
        intent = (self.batch["intent"] + flip) % len(INTENTS)
        slots = (self.batch["slots"].clamp(min=0) + flip[:, None]) % (CFG.max_tools + 1)
        labels = self.batch["dec_labels"].clamp(min=0)
        tokens = (labels + flip[:, None]) % CFG.vocab_size
        return Output(
            questions=Questions(
                intent=F.one_hot(intent, len(INTENTS)).float() * self.scale,
                slots=F.one_hot(slots, CFG.max_tools + 1).float() * self.scale,
            ),
            lm=F.one_hot(tokens, CFG.vocab_size).float() * self.scale,
        )


def sure(batch, scale, wrong=()):
    model = Sure(scale, wrong)
    model.batch = batch
    return model


def test_tools_count_the_slots_up_to_the_first_stop(batch):
    answered = _answered(
        sure(batch, 1.0)(*[batch[k] for k in ("enc_ids", "enc_mask")]), batch, CFG.max_tools
    )
    # 1 call + STOP, 2 calls + STOP, 1 call + STOP, unclear ASK (no labels), STOP, STOP
    assert answered["tools"].labels.numel() == 2 + 3 + 2 + 0 + 1 + 1
    assert answered["intent"].labels.tolist() == batch["intent"].tolist()
    values, _ = value_hits(
        F.one_hot(batch["dec_labels"].clamp(min=0), CFG.vocab_size), batch["dec_labels"]
    )
    assert int(answered["values"].group.max()) + 1 == int(values.sum())
    assert (answered["values"].labels != IGNORE).all()


def test_fit_spreads_out_an_overconfident_head(batch):
    """Half the intents wrong at logit 4 (the examples without values): the best temperature
    makes the answer's probability 1/2, e^(4/T) / (e^(4/T) + 3) = 1/2, so T = 4 / ln 3."""
    cal = fit(sure(batch, 4.0, wrong=(3, 4, 5)), [batch])
    assert cal.intent == pytest.approx(4 / math.log(3), rel=0.03)
    assert cal.values == GRID[0]  # always right: as sure as the grid allows

    right_only = fit(sure(batch, 4.0, wrong=(3, 4, 5)), [batch], [torch.arange(6) < 3])
    assert right_only.intent == GRID[0]  # the wrong examples left out


def test_reliability_before_and_after(batch):
    model = sure(batch, 4.0, wrong=(3, 4, 5))
    task = TaskSet([batch], [torch.ones(6, dtype=torch.bool)])  # all familiar
    cooled = Calibrations(familiar=Calibration(intent=4 / math.log(3)))
    raw, after = reliability(model, task, [Calibrations(), cooled])
    assert raw["intent"].right.tolist() == after["intent"].right.tolist()  # same answers
    assert raw["intent"].accuracy == 0.5
    assert raw["intent"].mean_confidence > 0.9 and raw["intent"].ece() > 0.4
    assert after["intent"].mean_confidence == pytest.approx(0.5, abs=1e-4)
    assert after["intent"].ece() == pytest.approx(0.0, abs=1e-4)


def test_each_example_gets_its_regimes_temperatures(batch):
    """The 3 right examples are familiar and keep T = 1; the 3 wrong ones are new and get a
    high temperature, so the model is unsure exactly where it is wrong."""
    model = sure(batch, 4.0, wrong=(3, 4, 5))
    task = TaskSet([batch], [torch.arange(6) < 3])
    assert task.counts() == {"familiar": 3, "new": 3}
    cal = Calibrations(new=Calibration(intent=8.0))
    (after,) = reliability(model, task, [cal])
    confidence = after["intent"].confidence
    assert (confidence[:3] > 0.9).all() and (confidence[3:] < 0.5).all()


def test_ece_of_a_calibrated_and_an_overconfident_head():
    calibrated = Reliability(torch.tensor([0.75] * 4), torch.tensor([True, True, True, False]))
    assert calibrated.ece() == pytest.approx(0.0)
    assert calibrated.bins(4)[3] == (4, pytest.approx(0.75), 0.75)
    sure_and_wrong = Reliability(torch.tensor([0.99, 0.99]), torch.tensor([True, False]))
    assert sure_and_wrong.ece() == pytest.approx(0.49)
    assert Reliability(torch.zeros(0), torch.zeros(0, dtype=torch.bool)).ece() == 0.0


def test_saved_temperatures_are_read_back(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    torch.manual_seed(0)
    model = Eve(CFG)
    save_checkpoint(tmp_path / "model.pt", model, tokenizer, {"task": "questions", "joint": True})
    assert Calibrations.from_meta(load_checkpoint(tmp_path / "model.pt").meta) == Calibrations()

    cal = Calibrations(Calibration(2.0, 1.5, 0.9), Calibration(6.0, 4.0, 1.2), frozenset({"ab"}))
    save_calibration(tmp_path / "model.pt", cal, {"familiar": "val", "new": "probe"})
    ckpt = load_checkpoint(tmp_path / "model.pt")
    assert Calibrations.from_meta(ckpt.meta) == cal
    assert ckpt.meta["calibrated_on"] == {"familiar": "val", "new": "probe"}
    assert ckpt.meta["joint"] is True
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(ckpt.model.state_dict()[name], weight)


def renamed(tool: Tool, description: str) -> Tool:
    return Tool.from_dict(tool.to_dict() | {"description": description})


def test_a_tool_is_familiar_only_with_the_same_schema(library, tmp_path):
    lights, music = library["set_lights"], library["play_music"]
    fan = Tool.from_dict({"name": "set_fan", "description": "Set the fan", "args": []})
    rows = [{"utterance": "hi", "tools": ["set_lights", fan.to_dict()], "intent": "none"}]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    known = known_tools(tmp_path / "train.jsonl", library)
    cal = Calibrations(Calibration(intent=1.0), Calibration(intent=5.0), known)
    assert cal.known == {lights.fingerprint(), fan.fingerprint()}
    assert cal.is_familiar([lights, fan]) and cal.for_tools([fan]).intent == 1.0
    assert not cal.is_familiar([lights, music])  # one new tool makes the request new
    assert cal.for_tools([lights, music]).intent == 5.0
    assert not cal.is_familiar([renamed(lights, "Switch the lights")])  # same name, new schema


def test_a_temperature_changes_the_probabilities_not_the_answers(library, tokenizer):
    torch.manual_seed(0)
    model = Eve(CFG)
    tools = [library["set_lights"], library["play_music"]]
    args = (model, tokenizer, "turn off the kitchen lights", tools, True)
    raw = answer_questions(*args)
    hot = answer_questions(*args, Calibration(intent=3.0, tools=3.0, values=3.0))
    assert max(hot.intent.values()) < max(raw.intent.values())
    assert max(hot.intent, key=hot.intent.get) == max(raw.intent, key=raw.intent.get)
    assert hot.calls == raw.calls
    for a, b in zip(raw.written, hot.written, strict=True):
        assert {k: v.value for k, v in a.values.items()} == {
            k: v.value for k, v in b.values.items()
        }


def _write_task(path: Path, library: dict[str, Tool]) -> Path:
    """Every set has the same examples; in probe-candidates, the first 3 call a new tool."""
    path.mkdir()
    exs = examples(library)
    rows = "".join(json.dumps(e.to_json()) + "\n" for e in exs)
    for name in ("train", "val", "heldout-templates", "massive-test"):
        (path / f"{name}.jsonl").write_text(rows)
    probe = [e.to_json(inline=True) for e in exs]
    for row in probe[:3]:
        row["tools"][0]["description"] = "A tool the model never saw"
    (path / "probe-candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in probe))
    return path


def test_eve_calibrate(
    tmp_path: Path,
    library: dict[str, Tool],
    tokenizer: BPETokenizer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "old.pt", Eve(CFG), tokenizer, {"task": "questions"})
    save_checkpoint(
        tmp_path / "joint.pt", Eve(CFG), tokenizer, {"task": "questions", "joint": True}
    )
    task = _write_task(tmp_path / "task", library)
    argv = ["calibrate", "--task", str(task), "--sets", "massive-test", "--device", "cpu"]

    with pytest.raises(SystemExit, match="joint"):
        main([*argv[:1], str(tmp_path / "old.pt"), *argv[1:]])

    main([*argv[:1], str(tmp_path / "joint.pt"), *argv[1:], "--json"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["fitted_on"] == {"familiar": "val", "new": "probe-candidates"}
    assert list(out["sets"]) == ["val", "probe-candidates", "massive-test"]
    assert out["sets"]["val"]["examples"] == {"familiar": 6, "new": 0}
    assert out["sets"]["probe-candidates"]["examples"] == {"familiar": 3, "new": 3}
    assert set(out["sets"]["val"]) == {"examples", "intent", "tools", "values"}
    assert set(out["sets"]["val"]["intent"]["before"]) == {"n", "accuracy", "confidence", "ece"}
    assert "calibration" not in load_checkpoint(tmp_path / "joint.pt").meta  # no --save

    main([*argv[:1], str(tmp_path / "joint.pt"), *argv[1:], "--save"])
    assert "temperatures saved" in capsys.readouterr().out
    saved = Calibrations.from_meta(load_checkpoint(tmp_path / "joint.pt").meta)
    assert saved.familiar == Calibration(**out["calibration"]["familiar"])
    assert saved.new == Calibration(**out["calibration"]["new"])
    assert saved.is_familiar([library["set_lights"]])
