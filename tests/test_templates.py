import numpy as np
import pytest

from colibri.data.generate.templates import (
    MAX_CALLS,
    check_template,
    fill,
    number_words,
    parse,
    tidy,
)
from colibri.data.schema import Arg, Tool

LIGHTS = Tool(
    "set_lights",
    "Turn lights on or off",
    (
        Arg("room", "enum", options=("kitchen", "living room")),
        Arg("on", "bool"),
        Arg("brightness", "int", required=False, min=0, max=100),
    ),
)
MESSAGE = Tool("send_message", "Send a text", (Arg("contact", "str"), Arg("text", "str")))
POOLS = {"contact": ("Anna", "my boss"), "text": ("I'm late",)}


def test_number_words():
    assert [number_words(n) for n in (0, 7, 13, 20, 45, 100, 307)] == [
        "zero",
        "seven",
        "thirteen",
        "twenty",
        "forty-five",
        "one hundred",
        "three hundred seven",
    ]
    with pytest.raises(ValueError):
        number_words(1000)


def test_parse():
    segments = parse("turn {on:on|off} the [{room} ]lights{brightness=40}")
    assert [s.optional for s in segments] == [False, True, False]
    assert segments[0].args == {"on"} and segments[1].args == {"room"}
    assert segments[2].args == {"brightness"}


@pytest.mark.parametrize(
    "template, problem",
    [
        ("turn on the lights", "never filled"),
        ("turn {on} the {room} lights", "needs"),
        ("turn {on:on|off} the {room:a|b} lights", "only bool"),
        ("turn {on:on|off} the {room} {room} lights", "twice"),
        ("{on:on|off} {room} {color}", "unknown"),
        ("{on:on|off} [{room} [x]]", "nested"),
        ("{on:on|off} {room} {brightness=400}", "above"),
    ],
)
def test_check_template_problems(template, problem):
    errors = check_template(LIGHTS, template, {})
    assert any(problem in e for e in errors), errors


def test_str_arguments_need_a_pool():
    assert check_template(MESSAGE, "text {contact}: {text}", POOLS) == []
    assert any("pool" in e for e in check_template(MESSAGE, "text {contact}: {text}", {}))


def test_fill():
    rng = np.random.default_rng(0)
    for _ in range(50):
        f = fill(LIGHTS, "turn {on:on|off} the [{room} ]lights[ to {brightness}%]", {}, rng)
        assert f.args["room"] in ("kitchen", "living room") and isinstance(f.args["on"], bool)
        assert f.text.startswith("turn on the " if f.args["on"] else "turn off the ")
        assert ("brightness" in f.args) == ("%" in f.text)
        assert "  " not in f.text and not f.text.endswith(" ")


def test_fixed_values_are_implied():
    f = fill(LIGHTS, "dim the {room} lights{on=true}{brightness=20}", {}, np.random.default_rng(0))
    assert f.args["on"] is True and f.args["brightness"] == 20
    assert f.implied == {"on", "brightness"}
    assert f.text.startswith("dim the ") and "20" not in f.text


def test_ask_drops_a_required_value():
    rng = np.random.default_rng(0)
    f = fill(LIGHTS, "turn {on:on|off} the [{room} ]lights", {}, rng, ask=True)
    assert f.args["room"] == "?" and f.text in ("turn on the lights", "turn off the lights")
    assert fill(LIGHTS, "turn {on:on|off} the {room} lights", {}, rng, ask=True) is None


def test_numbers_are_sometimes_words():
    rng = np.random.default_rng(0)
    texts = [fill(LIGHTS, "{room} {on=true} at {brightness}", {}, rng).text for _ in range(200)]
    assert any(t.split()[-1].isdigit() for t in texts)
    assert any(t.split()[-1].isalpha() for t in texts)


def test_tidy():
    assert tidy("  set it  to 40 % , please ") == "set it to 40%, please"


def test_fan_out_repeats_the_call_per_option():
    f = fill(LIGHTS, "turn {on:on|off} the lights everywhere{room=*}", {}, np.random.default_rng(0))
    assert f.fanout == "room" and "room" not in f.args
    calls = f.expand(LIGHTS)
    assert [c["room"] for c in calls] == ["kitchen", "living room"]  # schema order
    assert all(c["on"] == f.args["on"] for c in calls)
    plain = fill(LIGHTS, "turn {on:on|off} the {room} lights", {}, np.random.default_rng(0))
    assert plain.fanout is None and plain.expand(LIGHTS) == [plain.args]


def test_fan_out_needs_a_required_enum():
    optional = Tool("vacuum", "Vacuum", (Arg("room", "enum", required=False, options=("a", "b")),))
    assert any("required enum" in e for e in check_template(optional, "vacuum{room=*}", {}))
    assert any(
        "required enum" in e for e in check_template(MESSAGE, "text {contact=*}: {text}", POOLS)
    )
    many = Tool("t", "x", (Arg("room", "enum", options=tuple("abcdefghij")),))
    assert any(f"at most {MAX_CALLS}" in e for e in check_template(many, "all{room=*}", {}))
