from pathlib import Path

import numpy as np
import pytest

from colibri.data.generate.build import Sources
from colibri.data.generate.sampler import ToolBank, sample
from colibri.data.generate.validate import check_example
from colibri.data.render import IGNORE, render

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def sources():
    return Sources.load(ROOT / "data/tools", ROOT / "data/generation")


@pytest.fixture(scope="module")
def bank(sources):
    return ToolBank(sources.train)


@pytest.mark.parametrize("intent", ["call", "ask", "unsupported", "none"])
def test_samples_are_valid(bank, sources, intent):
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = sample(bank, rng, intent, chitchat=sources.chitchat, unsupported=sources.unsupported)
        ex = s.example
        assert ex.intent == intent and 1 <= len(ex.tools) <= 5
        assert check_example(ex, s.implied) == []
        caps = [bank.specs[bank.index[t.name]].capability for t in ex.tools]
        assert len(set(caps)) == len(caps)  # never two tools that do the same thing
        if intent == "ask":
            assert s.missing and all(ex.calls[0].args[m] == "?" for m in s.missing)


def test_unsupported_never_offers_a_tool_that_could_serve_it(bank, sources):
    rng = np.random.default_rng(1)
    for _ in range(300):
        s = sample(bank, rng, "unsupported")
        names = {t.name for t in s.example.tools}
        assert "search_web" not in names  # it could serve anything


def test_distractors_are_mostly_similar(bank: ToolBank) -> None:
    rng = np.random.default_rng(0)
    gold = bank.index["set_lights"]
    sims = [bank.similarity[gold, d] for _ in range(200) for d in bank.distractors([gold], 4, rng)]
    random = bank.similarity[gold][np.arange(len(bank.specs)) != gold].mean()
    assert np.mean(sims) > 3 * random


def test_multi_call_examples(bank):
    rng = np.random.default_rng(0)
    multi = [
        s.example
        for s in (sample(bank, rng, "call") for _ in range(400))
        if len(s.example.calls) > 1
    ]
    assert len(multi) > 40
    assert any(len({c.tool for c in ex.calls}) < len(ex.calls) for ex in multi)  # same tool twice


def test_held_out_pool(sources):
    bank = ToolBank(sources.train + sources.heldout)
    pool = [bank.index[s.name] for s in sources.heldout]
    rng = np.random.default_rng(0)
    for _ in range(100):
        ex = sample(bank, rng, "call", pool=pool).example
        assert all(c.tool in {s.name for s in sources.heldout} for c in ex.calls)


def test_conversions_are_sensible(sources):
    bank = ToolBank(sources.heldout)
    sides = {
        "convert_units": ("from_unit", "to_unit"),
        "convert_currency": ("from_currency", "to_currency"),
    }
    units = {("km", "miles"), ("kg", "pounds"), ("celsius", "fahrenheit")}
    units |= {(b, a) for a, b in units}
    pool = [bank.index[name] for name in sides]
    rng = np.random.default_rng(0)
    seen = {name: set() for name in sides}
    for _ in range(1000):
        for c in sample(bank, rng, "call", pool=pool).example.calls:
            seen[c.tool].add(tuple(c.args[side] for side in sides[c.tool]))
    assert seen["convert_units"] == units  # every same-kind pair, nothing else
    assert len(seen["convert_currency"]) == 4 * 3  # every pair of different currencies
    assert all(a != b for a, b in seen["convert_currency"])
    missing = {name: set() for name in sides}
    for _ in range(300):
        s = sample(bank, rng, "ask", pool=pool)
        missing[s.example.calls[0].tool] |= set(s.missing)
    assert missing == {
        "convert_units": {"value", "to_unit"},  # "convert km to miles", "convert 5 km"
        "convert_currency": {"amount", "to_currency"},
    }


def test_samples_render(bank, tokenizer):
    from colibri.config import preset

    rng = np.random.default_rng(0)
    for _ in range(50):
        r = render(sample(bank, rng).example, tokenizer, preset("tiny"), rng)
        assert any(label != IGNORE for label in r.slots) or r.intent == 1


def test_fan_out_examples(bank):
    from colibri.data.generate.templates import MAX_CALLS

    rng = np.random.default_rng(0)
    fanned = []
    for _ in range(3000):
        ex = sample(bank, rng, "call").example
        assert len(ex.calls) <= MAX_CALLS
        if ex.calls[0].tool == "set_lights" and "everywhere" in ex.utterance:
            fanned.append(ex)
    assert fanned
    rooms = bank.specs[bank.index["set_lights"]].tool.arg("room").options
    for ex in fanned:
        assert [c.args["room"] for c in ex.calls[: len(rooms)]] == list(rooms)
