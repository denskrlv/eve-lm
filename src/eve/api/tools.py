"""Tools for the public API, from three sources, all turned into the ``Tool`` the model reads:

- a typed Python function (``@eve.tool``): its name, docstring and type hints;
- an OpenAI- or MCP-style JSON Schema: ``{"type": "function", "function": {...}}``,
  ``{"name", "description", "parameters"}``, or ``input_schema`` / ``inputSchema``;
- Eve's own schema (``{"name", "description", "args": [...]}``, as in ``data/tools``) or a
  ``Tool``.

Eve's arguments are scalars: str, int, float, bool or an enum of strings. Anything else (a list,
an object, a union of two types) raises a ``ToolError`` that names the argument.
"""

import inspect
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from eve.api.errors import ToolError
from eve.data.schema import GENERATION_KEYS, Arg, SchemaError, Tool

TOOL_ATTR = "__eve_tool__"  # where ``@tool`` keeps the schema it read
PYTHON_TYPES: dict[Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}
JSON_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
SCALARS = "str, int, float, bool or an enum of strings"
_ARGS_HEADER = re.compile(r"^(args|arguments|parameters):\s*$", re.I)
_ARG_LINE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")  # "room (str): the room"

ToolLike = Callable[..., Any] | Mapping[str, Any] | Tool


@dataclass(frozen=True)
class Param:
    """More about one argument of a function tool, given with ``Annotated``:
    ``brightness: Annotated[int, Param(min=0, max=100, description="percent")]``."""

    description: str = ""
    min: float | None = None
    max: float | None = None


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _tool(name: str, build: Callable[[], Tool]) -> Tool:
    """``build()``, with a schema rule it breaks raised as a ``ToolError``."""
    try:
        return build()
    except SchemaError as error:
        raise ToolError(f"tool {name!r}: {error}") from None


# ---- Python functions -------------------------------------------------------------------------


def _docstring(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """The description (the docstring's first paragraph) and each argument's description from
    a Google-style ``Args:`` section."""
    doc = inspect.getdoc(fn) or ""
    description = _one_line(doc.split("\n\n")[0]) if doc else ""
    args: dict[str, str] = {}
    current, inside, base = "", False, -1
    for line in doc.splitlines():
        if _ARGS_HEADER.match(line.strip()):
            inside = True
            continue
        if not inside or not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:  # the next section ("Returns:") ends it
            break
        if base < 0:
            base = indent
        if indent == base and (match := _ARG_LINE.match(line.strip())):
            current = match[1]
            args[current] = match[2]
        elif current:  # a description going on over more lines
            args[current] += " " + line.strip()
    return description, {name: _one_line(text) for name, text in args.items()}


def _python_arg(name: str, hint: Any, has_default: bool, described: str) -> Arg:
    """One argument from its type hint: ``Annotated`` adds a ``Param``, ``X | None`` or a
    default makes it optional, ``Literal`` of strings is an enum."""
    param, required = Param(), not has_default
    for _ in range(2):  # Annotated[X | None, ...] and Annotated[X, ...] | None alike
        if get_origin(hint) is Annotated:
            hint, *extras = get_args(hint)
            param = next((e for e in extras if isinstance(e, Param)), param)
        if get_origin(hint) in (Union, types.UnionType):
            options = [h for h in get_args(hint) if h is not type(None)]
            if len(options) != 1:
                raise ToolError(f"argument {name!r} is {hint}; Eve's arguments are {SCALARS}")
            hint, required = options[0], False
    description = param.description or described
    if get_origin(hint) is Literal:
        values = get_args(hint)
        if not all(isinstance(v, str) for v in values):
            raise ToolError(f"argument {name!r}: a Literal must hold strings, got {values}")
        return Arg(name, "enum", required, tuple(values), description=description)
    if hint not in PYTHON_TYPES:
        raise ToolError(f"argument {name!r} is {hint}; Eve's arguments are {SCALARS}")
    return Arg(name, PYTHON_TYPES[hint], required, (), param.min, param.max, description)


def from_function(fn: Callable[..., Any]) -> Tool:
    """The schema of a typed Python function: its name, its docstring's first paragraph as the
    description, and one argument per parameter."""
    name = getattr(fn, "__name__", repr(fn))
    description, described = _docstring(fn)
    if not description:
        raise ToolError(f"tool {name!r} has no docstring; Eve reads it as the description")
    try:
        hints = get_type_hints(fn, include_extras=True)
    except (NameError, TypeError) as error:
        raise ToolError(f"tool {name!r}: can't read its type hints ({error})") from None
    args = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ToolError(f"tool {name!r}: *args and **kwargs can't be described to Eve")
        if param.name not in hints:
            raise ToolError(f"tool {name!r}: argument {param.name!r} has no type hint")
        has_default = param.default is not param.empty
        try:
            args.append(_python_arg(param.name, hints[param.name], has_default,
                                    described.get(param.name, "")))  # fmt: skip
        except (SchemaError, ToolError) as error:
            raise ToolError(f"tool {name!r}: {error}") from None
    return _tool(name, lambda: Tool(name, description, tuple(args)))


def tool[F: Callable[..., Any]](fn: F) -> F:
    """Mark ``fn`` as a tool Eve can call. Its schema is read here, so a hint Eve can't use
    fails when the module is imported, not on the first request::

        @eve.tool
        def set_lights(room: str, on: bool, brightness: int | None = None):
            \"\"\"Turn the lights in a room on or off.\"\"\"
    """
    setattr(fn, TOOL_ATTR, from_function(fn))
    return fn


# ---- JSON Schema (OpenAI, MCP) ----------------------------------------------------------------


def _json_type(name: str, spec: Mapping[str, Any]) -> tuple[str, bool]:
    """The JSON type of a property and whether it may be null (``["string", "null"]``,
    ``anyOf`` with ``{"type": "null"}``)."""
    kind = spec.get("type")
    if kind is None and "anyOf" in spec:
        kinds = [s.get("type") for s in spec["anyOf"] if isinstance(s, Mapping)]
        kind = kinds
    if isinstance(kind, list):
        rest = [k for k in kind if k != "null"]
        if len(rest) != 1:
            raise ToolError(f"argument {name!r} has types {kind}; Eve's arguments are {SCALARS}")
        return rest[0], len(rest) < len(kind)
    if kind is None and "enum" in spec:
        return "string", False
    if not isinstance(kind, str):
        raise ToolError(f"argument {name!r} has no type; Eve's arguments are {SCALARS}")
    return kind, False


def _json_arg(name: str, spec: Mapping[str, Any], required: bool) -> Arg:
    kind, nullable = _json_type(name, spec)
    required = required and not nullable
    description = _one_line(str(spec.get("description", "")))
    if "enum" in spec:
        values = [v for v in spec["enum"] if v is not None]
        if kind != "string" or not all(isinstance(v, str) for v in values):
            raise ToolError(f"argument {name!r}: Eve's enums hold strings, got {spec['enum']}")
        return Arg(name, "enum", required, tuple(values), description=description)
    if kind not in JSON_TYPES:
        raise ToolError(f"argument {name!r} is an {kind}; Eve's arguments are {SCALARS}")
    low, high = spec.get("minimum"), spec.get("maximum")
    return Arg(name, JSON_TYPES[kind], required, (), low, high, description)


def from_json_schema(data: Mapping[str, Any]) -> Tool:
    """A tool from an OpenAI function, an Anthropic or MCP tool, or a bare
    ``{"name", "description", "parameters"}``."""
    if data.get("type") == "function" and isinstance(data.get("function"), Mapping):
        data = data["function"]
    name = str(data.get("name", ""))
    schema = data.get("parameters") or data.get("input_schema") or data.get("inputSchema") or {}
    if schema.get("type", "object") != "object":
        raise ToolError(f"tool {name!r}: its parameters must be an object schema")
    required = set(schema.get("required", ()))
    args = []
    for arg_name, spec in schema.get("properties", {}).items():
        try:
            args.append(_json_arg(arg_name, spec, arg_name in required))
        except (SchemaError, ToolError) as error:
            raise ToolError(f"tool {name!r}: {error}") from None
    description = _one_line(str(data.get("description", "")))
    return _tool(name, lambda: Tool(name, description, tuple(args)))


# ---- any of them ------------------------------------------------------------------------------


def to_tool(source: ToolLike) -> tuple[Tool, Callable[..., Any] | None]:
    """The ``Tool`` for ``source``, and the function to call if it is one."""
    if isinstance(source, Tool):
        return source, None
    if isinstance(source, Mapping):
        if "args" in source and not {"parameters", "input_schema", "inputSchema"} & set(source):
            data = {k: v for k, v in source.items() if k not in GENERATION_KEYS}
            name = str(source.get("name", ""))
            return _tool(name, lambda: Tool.from_dict(data)), None
        return from_json_schema(source), None
    if callable(source):
        found = getattr(source, TOOL_ATTR, None)
        return (found if isinstance(found, Tool) else from_function(source)), source
    raise ToolError(f"{source!r} is not a tool: give a function, a JSON schema dict or a Tool")
