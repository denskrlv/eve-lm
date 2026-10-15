from colibri.data.generate.validate import check_example, mentions, normalize, value_forms
from colibri.data.render import Call, TaskExample
from colibri.data.schema import Arg, Tool

TIMER = Tool(
    "set_timer",
    "Start a timer",
    (Arg("minutes", "int", min=1), Arg("label", "str", required=False)),
)
LIGHTS = Tool(
    "set_lights", "Lights", (Arg("room", "enum", options=("kitchen", "office")), Arg("on", "bool"))
)


def ex(text, intent="call", **args):
    calls = (Call("set_timer", args),) if args else ()
    return TaskExample(text, (TIMER, LIGHTS), intent, calls)


def test_values_must_appear():
    assert check_example(ex("set a pasta timer for 10 minutes", minutes=10, label="pasta")) == []
    assert check_example(ex("set a timer for ten minutes", minutes=10)) == []
    assert check_example(ex("timer, twenty five minutes", minutes=25)) == []
    assert check_example(ex("set a timer for 15 minutes", minutes=10))
    assert check_example(ex("set a timer for 100 minutes", minutes=10))  # not a whole word
    assert check_example(ex("set an egg timer for 10 minutes", minutes=10, label="pasta"))


def test_bools_enums_and_implied_values_are_exempt():
    lights = TaskExample(
        "kill the lights in the study",
        (LIGHTS,),
        "call",
        (Call("set_lights", {"room": "office", "on": False}),),
    )
    assert check_example(lights) == []
    implied = ex("a quick timer please", minutes=5)
    assert check_example(implied) and check_example(implied, {(0, "minutes")}) == []


def test_ask_needs_a_missing_value():
    assert check_example(ex("set a timer", "ask", minutes="?")) == []
    assert check_example(ex("set a timer for 5 minutes", "ask", minutes=5))


def test_helpers():
    assert value_forms(Arg("t", "float"), 21.0) == ["21", "twenty-one", "twenty one"]
    assert mentions("Call My Boss now", "my boss") and not mentions("bossy", "boss")
    assert normalize("Turn OFF, the lights!") == "turn off the lights"
