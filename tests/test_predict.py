import json
from pathlib import Path

import pytest
import torch

from eve.cli import main
from eve.config import INTENTS, preset
from eve.data.schema import Tool
from eve.infer.predict import KINDS, STOP, Answers, answer_questions, write_values
from eve.model.checkpoint import save_checkpoint
from eve.model.eve import Eve
from eve.tokenizer.bpe import ABSENT, ASK, END, BPETokenizer

CFG = preset("tiny")


class Scripted(Eve):
    """A model whose decoder writes ``script``, one token per call, whatever it reads."""

    def __init__(self, script):
        super().__init__(CFG)
        self.script = list(script)

    def decode(self, dec_ids, *args, **kwargs):
        logits = torch.zeros(1, dec_ids.shape[1], CFG.vocab_size)
        logits[0, -1, self.script.pop(0)] = 10.0
        return logits


def write(tokenizer, library, script, tools=("set_lights",)):
    model = Scripted(script)
    enc = torch.ones(1, 4, dtype=torch.bool)
    memory = torch.zeros(1, 4, CFG.d_model)
    out = write_values(model, tokenizer, memory, enc, [library[t] for t in tools])
    assert not model.script  # it wrote every scripted token, and no more
    return out


def text(tokenizer, value):
    return tokenizer.encode(" " + value)


def test_answers_are_distributions_over_the_prompts_tools(library, tokenizer):
    torch.manual_seed(0)
    model = Eve(preset("tiny")).train()
    tools = [library["set_lights"], library["play_music"]]
    a = answer_questions(model, tokenizer, "turn off the kitchen lights", tools)
    assert list(a.intent) == list(INTENTS)
    assert sum(a.intent.values()) == pytest.approx(1.0)
    assert len(a.slots) == preset("tiny").n_slots
    for slot in a.slots:
        assert list(slot) == ["set_lights", "play_music", STOP]  # padded tools left out
        assert sum(slot.values()) == pytest.approx(1.0, abs=1e-5)
    assert model.training  # put back the way it was


def test_calls_stop_at_the_first_stop():
    a = Answers(
        intent={"call": 1.0},
        slots=[{"a": 0.9, STOP: 0.1}, {"a": 0.2, STOP: 0.8}, {"a": 0.9, STOP: 0.1}],
    )
    assert a.calls == ["a"]


def test_answers_copy_the_batch_blocking(
    library: dict[str, Tool], tokenizer: BPETokenizer, copy_flags: list[bool]
) -> None:
    """The collated batch is freed right after the copy: a non-blocking copy could still be
    reading it on MPS."""
    model = Eve(preset("tiny"))
    answer_questions(model, tokenizer, "turn off the lights", [library["set_lights"]])
    assert copy_flags and not any(copy_flags)


def test_values_are_parsed_into_the_arguments_types(library, tokenizer):
    script = [*text(tokenizer, "living room"), END, *text(tokenizer, "false"), END, ABSENT]
    (values,) = write(tokenizer, library, script)
    assert {name: v.kind for name, v in values.items()} == {
        "room": "value", "on": "value", "brightness": "absent"
    }  # fmt: skip
    assert values["room"].value == "living room" and values["on"].value is False
    assert all(0.0 < v.prob <= 1.0 for v in values.values())


def test_a_missing_value_and_an_invalid_one(library, tokenizer):
    script = [ASK, *text(tokenizer, "maybe"), END, *text(tokenizer, "30"), END]
    (values,) = write(tokenizer, library, script)
    assert values["room"].kind == "ask"
    assert values["on"].kind == "invalid" and values["on"].value == "maybe"
    assert values["brightness"].kind == "value" and values["brightness"].value == 30


def test_each_call_gets_its_own_values(library, tokenizer):
    lights = [*text(tokenizer, "kitchen"), END, *text(tokenizer, "true"), END, ABSENT]
    thermostat = [*text(tokenizer, "21.5"), END, ABSENT]
    first, second = write(
        tokenizer, library, lights + thermostat, tools=("set_lights", "set_thermostat")
    )
    assert first["room"].value == "kitchen" and second["temperature"].value == 21.5


def test_decoding_follows_the_renderers_rules(library, tokenizer):
    """Whatever an untrained decoder prefers: a required argument is never <absent>, an
    optional one never <ask>, and a value never starts with <end>."""
    torch.manual_seed(0)
    model = Eve(CFG).eval()
    tools = [library["set_lights"], library["set_thermostat"]]
    for _ in range(3):
        answers = answer_questions(model, tokenizer, "turn off the lights", tools, values=True)
        for call in answers.written:
            tool = library[call.tool]
            assert list(call.values) == [arg.name for arg in tool.args]
            for name, value in call.values.items():
                assert value.kind in KINDS
                kind = "ask" if tool.arg(name).required else "absent"
                assert value.kind != {"ask": "absent", "absent": "ask"}[kind]
        torch.nn.init.normal_(model.lm_head.weight)  # another random decoder


def test_answers_write_a_call_per_chosen_tool(library, tokenizer):
    torch.manual_seed(0)
    model = Eve(CFG)
    tools = [library["set_lights"], library["play_music"]]
    answers = answer_questions(model, tokenizer, "turn off the lights", tools, values=True)
    assert [c.tool for c in answers.written] == answers.calls
    assert answer_questions(model, tokenizer, "turn off the lights", tools).written == []
    calls = answers.to_json()["calls"]
    assert all({"tool", "args", "missing", "prob", "values"} <= set(c) for c in calls)


def test_eve_ask_writes_values_only_for_a_joint_model(
    tmp_path: Path, tokenizer: BPETokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.manual_seed(0)  # its slot head picks set_lights, then STOP
    model = Eve(CFG)
    save_checkpoint(tmp_path / "joint.pt", model, tokenizer, {"task": "questions", "joint": True})
    save_checkpoint(tmp_path / "old.pt", model, tokenizer, {"task": "questions"})
    argv = ["ask", "turn off the kitchen lights", "--tools", "set_lights", "--json"]

    main([*argv, "--model", str(tmp_path / "old.pt"), "--device", "cpu"])
    old = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert old["calls"] == ["set_lights"]  # tool names only

    main([*argv, "--model", str(tmp_path / "joint.pt"), "--device", "cpu"])
    joint = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    (call,) = joint["calls"]
    assert call["tool"] == "set_lights" and set(call["values"]) == {"room", "on", "brightness"}
