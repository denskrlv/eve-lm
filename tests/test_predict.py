import pytest
import torch

from colibri.config import INTENTS, preset
from colibri.infer.predict import STOP, Answers, answer_questions
from colibri.model.colibri import Colibri


def test_answers_are_distributions_over_the_prompts_tools(library, tokenizer):
    torch.manual_seed(0)
    model = Colibri(preset("tiny")).train()
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
