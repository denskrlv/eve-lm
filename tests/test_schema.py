import json

import pytest

from eve.data.schema import Arg, SchemaError, Tool, load_library, load_tools


def test_library_loads(library):
    assert len(library) >= 15
    assert {t.domain for t in library.values()} >= {"home", "media", "time"}
    for tool in library.values():
        assert tool.render().startswith(f"<tool> {tool.name} : ")


def test_render(library):
    assert library["set_lights"].render() == (
        "<tool> set_lights : Turn the lights in a room on or off"
        " <arg> room : enum kitchen | bedroom | living room | bathroom | office | hallway"
        " <arg> on : bool"
        " <arg> brightness ? : int 0..100 : percent"
    )
    assert Arg("t", "float", min=10, max=30.5).render() == "<arg> t : float 10..30.5"
    assert Arg("n", "int", min=1).render() == "<arg> n : int 1.."
    assert library["pause_media"].render() == "<tool> pause_media : Pause whatever is playing"


def test_round_trip_through_dict(library):
    for tool in library.values():
        assert Tool.from_dict(tool.to_dict()) == tool


@pytest.mark.parametrize(
    "make",
    [
        lambda: Arg("Room", "str"),
        lambda: Arg("room", "string"),
        lambda: Arg("room", "enum", options=("kitchen",)),
        lambda: Arg("room", "enum", options=("a", "a")),
        lambda: Arg("room", "enum", options=("a", "b|c")),
        lambda: Arg("room", "str", options=("a", "b")),
        lambda: Arg("level", "str", min=0),
        lambda: Arg("level", "int", min=5, max=1),
        lambda: Arg("level", "int", min=0.5),
        lambda: Arg("note", "str", description="see <tool> below"),
        lambda: Tool("set lights", "x"),
        lambda: Tool("set_lights", " "),
        lambda: Tool("set_lights", "x", (Arg("a", "str"), Arg("a", "int"))),
        lambda: Tool("t", "x", tuple(Arg(f"a{i}", "str") for i in range(9))),
        lambda: Tool.from_dict({"name": "t", "description": "x", "argz": []}),
        lambda: Arg.from_dict({"name": "a", "type": "str", "optional": True}),
    ],
)
def test_invalid_schemas_are_rejected(make):
    with pytest.raises(SchemaError):
        make()


def test_check_call(library):
    lights = library["set_lights"]
    assert lights.check_call({"room": "kitchen", "on": False}) == []
    assert lights.check_call({"room": "kitchen", "on": True, "brightness": 40}) == []
    assert lights.check_call({"room": "garden", "on": True})  # not an option
    assert lights.check_call({"room": "kitchen"})  # missing required
    assert lights.check_call({"room": "kitchen", "on": 1})  # int is not bool
    assert lights.check_call({"room": "kitchen", "on": True, "brightness": 140})  # out of range
    assert lights.check_call({"room": "kitchen", "on": True, "color": "red"})  # unknown
    assert lights.check_call({"room": "?", "on": True})
    assert lights.check_call({"room": "?", "on": True}, allow_ask=True) == []
    thermostat = library["set_thermostat"]
    assert thermostat.check_call({"temperature": 21}) == []  # an int is a fine float
    assert thermostat.check_call({"temperature": True})


def test_format(library):
    assert library["set_lights"].arg("on").format(False) == "false"
    assert library["set_thermostat"].arg("temperature").format(21.5) == "21.5"
    assert library["set_thermostat"].arg("temperature").format(20.0) == "20"
    assert library["set_lights"].arg("room").format("living room") == "living room"


def test_duplicate_tool_names_across_files(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"domain": name, "tools": [{"name": "same", "description": "x"}]})
        )
    assert load_tools(tmp_path / "a.json")[0].domain == "a"
    with pytest.raises(SchemaError):
        load_library(tmp_path)
