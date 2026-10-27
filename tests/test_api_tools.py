from typing import Annotated, Literal

import pytest

import eve
from eve.api.tools import Param, from_function, from_json_schema, to_tool, tool
from eve.data.schema import Arg, Tool


def lights(
    room: str,
    on: bool,
    brightness: Annotated[int, Param(min=0, max=100)] | None = None,
    mode: Literal["warm", "cold"] = "warm",
    speed: float = 1.0,
) -> str:
    """Turn the lights in a room on or off.

    Not part of the description.

    Args:
        room: Which room, e.g. the
            kitchen.
        on (bool): True to turn them on.

    Returns:
        A message.
    """
    return f"{room} {on}"


def test_a_typed_function_becomes_a_schema() -> None:
    schema = from_function(lights)
    assert schema.name == "lights"
    assert schema.description == "Turn the lights in a room on or off."
    assert schema.args == (
        Arg("room", "str", description="Which room, e.g. the kitchen."),
        Arg("on", "bool", description="True to turn them on."),
        Arg("brightness", "int", required=False, min=0, max=100),
        Arg("mode", "enum", required=False, options=("warm", "cold")),
        Arg("speed", "float", required=False),
    )


def test_the_decorator_reads_the_schema_once_and_keeps_the_function() -> None:
    decorated = tool(lights)
    assert decorated is lights and to_tool(decorated) == (from_function(lights), lights)
    assert eve.tool is tool and eve.Param is Param


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("def f(rooms: list[str]):\n    '''Do it.'''", "Eve's arguments are str, int"),
        ("def f(x: int | str):\n    '''Do it.'''", "Eve's arguments are str, int"),
        ("def f(x: Literal[1, 2]):\n    '''Do it.'''", "a Literal must hold strings"),
        ("def f(room):\n    '''Do it.'''", "has no type hint"),
        ("def f(*rooms: str):\n    '''Do it.'''", r"\*args and \*\*kwargs"),
        ("def f(room: str):\n    pass", "has no docstring"),
        ("def setLights(room: str):\n    '''Do it.'''", "snake_case"),
    ],
)
def test_functions_eve_cant_read_say_why(source: str, reason: str) -> None:
    namespace: dict[str, object] = {"Literal": Literal}
    exec(source, namespace)
    function = next(v for k, v in namespace.items() if callable(v) and k != "Literal")
    with pytest.raises(eve.ToolError, match=reason):
        from_function(function)  # type: ignore[arg-type]


OPENAI = {
    "type": "function",
    "function": {
        "name": "set_thermostat",
        "description": "Set the target temperature.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "the room"},
                "celsius": {"type": "number", "minimum": 5, "maximum": 30},
                "mode": {"type": "string", "enum": ["heat", "cool"]},
                "eco": {"type": ["boolean", "null"]},
                "hours": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["room", "celsius", "eco"],
        },
    },
}


def test_openai_and_mcp_schemas_become_the_same_tool() -> None:
    schema = from_json_schema(OPENAI)
    assert schema.args == (
        Arg("room", "str", description="the room"),
        Arg("celsius", "float", min=5, max=30),
        Arg("mode", "enum", required=False, options=("heat", "cool")),
        Arg("eco", "bool", required=False),  # nullable, so it may be left out
        Arg("hours", "int", required=False),
    )
    function = OPENAI["function"]
    assert isinstance(function, dict)
    mcp = {"name": function["name"], "description": function["description"],
           "inputSchema": function["parameters"]}  # fmt: skip
    assert to_tool(mcp) == (schema, None)


@pytest.mark.parametrize(
    ("prop", "reason"),
    [
        ({"type": "array", "items": {"type": "string"}}, "is an array"),
        ({"type": "object"}, "is an object"),
        ({"type": ["string", "integer"]}, "has types"),
        ({"type": "integer", "enum": [1, 2]}, "enums hold strings"),
        ({"description": "no type"}, "has no type"),
    ],
)
def test_json_schemas_eve_cant_read_say_why(prop: dict[str, object], reason: str) -> None:
    data = {"name": "f", "description": "Do it.", "parameters": {"properties": {"x": prop}}}
    with pytest.raises(eve.ToolError, match=reason):
        from_json_schema(data)


def test_eve_schemas_and_tools_pass_through(library: dict[str, Tool]) -> None:
    lamp = library["set_lights"]
    assert to_tool(lamp) == (lamp, None)
    data = lamp.to_dict() | {"templates": ["turn off the {room} lights"]}  # generation keys
    assert to_tool(data) == (lamp, None)
    with pytest.raises(eve.ToolError, match="unknown keys"):
        to_tool(lamp.to_dict() | {"colour": "red"})
    with pytest.raises(eve.ToolError, match="is not a tool"):
        to_tool(42)  # type: ignore[arg-type]
