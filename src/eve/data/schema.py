import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eve import defaults
from eve.tokenizer.bpe import SPECIAL_TOKENS

ARG_TYPES = ("str", "int", "float", "bool", "enum")
MAX_ARGS = 8  # per tool; the renderer returns one <arg> position per argument, padded to this
ASK_VALUE = "?"  # in task data: a required value the user didn't give (intent "ask")

_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")  # snake_case


class SchemaError(ValueError):
    """A tool schema, call or example that breaks the rules."""


def _check_text(what: str, text: str) -> None:
    """Free text must not contain a special token, or it would add a fake <tool>/<arg> marker."""
    for special in SPECIAL_TOKENS:
        if special in text:
            raise SchemaError(f"{what} contains the special token {special}: {text!r}")


def _check_name(what: str, name: str) -> None:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise SchemaError(f"{what} must be snake_case, got {name!r}")


def _number(value: float) -> str:
    """16 -> "16", 21.5 -> "21.5" (never "16.0" or "2e+01")."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


@dataclass(frozen=True)
class Arg:
    """One argument of a tool: its type, whether it is required, its options or range."""

    name: str
    type: str
    required: bool = True
    options: tuple[str, ...] = ()  # enum only
    min: float | None = None  # int and float only
    max: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _check_name("argument name", self.name)
        _check_text(f"argument {self.name}", self.description)
        if self.type not in ARG_TYPES:
            raise SchemaError(f"argument {self.name}: type must be one of {ARG_TYPES}")

        if self.type == "enum":
            if len(self.options) < 2 or len(set(self.options)) != len(self.options):
                raise SchemaError(f"argument {self.name}: an enum needs 2+ distinct options")
            for option in self.options:
                if not option or option != option.strip() or "|" in option:
                    raise SchemaError(f"argument {self.name}: bad enum option {option!r}")
                _check_text(f"argument {self.name}", option)
        elif self.options:
            raise SchemaError(f"argument {self.name}: only enums have options")

        if self.type in ("int", "float"):
            for bound in (self.min, self.max):
                if bound is not None and self.type == "int" and not float(bound).is_integer():
                    raise SchemaError(f"argument {self.name}: int bounds must be whole numbers")
            if self.min is not None and self.max is not None and self.min > self.max:
                raise SchemaError(f"argument {self.name}: min > max")
        elif self.min is not None or self.max is not None:
            raise SchemaError(f"argument {self.name}: only int and float have a range")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Arg":
        """An argument from its JSON form, as in ``data/tools``."""
        unknown = set(data) - {"name", "type", "required", "options", "min", "max", "description"}
        if unknown:
            raise SchemaError(f"argument {data.get('name')!r}: unknown keys {sorted(unknown)}")
        return cls(**{**data, "options": tuple(data.get("options", ()))})

    def to_dict(self) -> dict[str, Any]:
        """The JSON form, without the fields that have their default."""
        out: dict[str, Any] = {"name": self.name, "type": self.type}
        if not self.required:
            out["required"] = False
        if self.options:
            out["options"] = list(self.options)
        for key in ("min", "max", "description"):
            if getattr(self, key) not in (None, ""):
                out[key] = getattr(self, key)
        return out

    def render(self) -> str:
        """The argument as the encoder reads it: ``<arg> name ? : type : note``."""
        spec = self.type
        if self.type == "enum":
            spec += " " + " | ".join(self.options)
        elif self.min is not None or self.max is not None:
            lo = "" if self.min is None else _number(self.min)
            hi = "" if self.max is None else _number(self.max)
            spec += f" {lo}..{hi}"
        text = f"<arg> {self.name}{'' if self.required else ' ?'} : {spec}"
        return text + (f" : {self.description}" if self.description else "")

    def check(self, value: Any) -> str | None:
        """Return why ``value`` is not valid for this argument, or None if it is."""
        if self.type == "str":
            ok = isinstance(value, str) and value.strip() != ""
        elif self.type == "int":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif self.type == "float":
            ok = isinstance(value, int | float) and not isinstance(value, bool)
            ok = ok and math.isfinite(value)
        elif self.type == "bool":
            ok = isinstance(value, bool)
        else:
            ok = value in self.options
        if not ok:
            return f"{self.name}: {value!r} is not a valid {self.type}"
        if self.type == "str":
            try:
                _check_text(self.name, value)
            except SchemaError as e:
                return str(e)
        if self.min is not None and value < self.min:
            return f"{self.name}: {value} is below {_number(self.min)}"
        if self.max is not None and value > self.max:
            return f"{self.name}: {value} is above {_number(self.max)}"
        return None

    def format(self, value: Any) -> str:
        """The value as the decoder writes it (M8): true/false, 21.5, kitchen."""
        if self.type == "bool":
            return "true" if value else "false"
        if self.type in ("int", "float"):
            return _number(value)
        text: str = value  # str and enum values are already text
        return text


@dataclass(frozen=True)
class Tool:
    """A tool the model can call: its name, description and arguments."""

    name: str
    description: str
    args: tuple[Arg, ...] = ()
    domain: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        _check_name("tool name", self.name)
        if not self.description.strip():
            raise SchemaError(f"tool {self.name}: description is empty")
        _check_text(f"tool {self.name}", self.description)
        names = [a.name for a in self.args]
        if len(set(names)) != len(names):
            raise SchemaError(f"tool {self.name}: duplicate argument names")
        if len(self.args) > MAX_ARGS:
            raise SchemaError(f"tool {self.name}: more than {MAX_ARGS} arguments")

    @classmethod
    def from_dict(cls, data: dict[str, Any], domain: str = "") -> "Tool":
        """A tool from its JSON form, as in ``data/tools``."""
        unknown = set(data) - {"name", "description", "args"}
        if unknown:
            raise SchemaError(f"tool {data.get('name')!r}: unknown keys {sorted(unknown)}")
        args = tuple(Arg.from_dict(a) for a in data.get("args", ()))
        return cls(data["name"], data["description"], args, domain)

    def to_dict(self) -> dict[str, Any]:
        """The JSON form."""
        return {
            "name": self.name,
            "description": self.description,
            "args": [a.to_dict() for a in self.args],
        }

    def render_parts(self) -> list[str]:
        """The header and one part per argument, each starting with its marker.

        The renderer encodes the parts one by one. Encoding the joined text would turn the space
        before each ``<arg>`` into a token of its own.
        """
        return [f"<tool> {self.name} : {self.description}"] + [a.render() for a in self.args]

    def render(self) -> str:
        """The whole schema as one string (see ``render_parts``)."""
        return " ".join(self.render_parts())

    def arg(self, name: str) -> Arg:
        """The argument called ``name``."""
        for arg in self.args:
            if arg.name == name:
                return arg
        raise KeyError(f"tool {self.name} has no argument {name!r}")

    def check_call(self, values: dict[str, Any], allow_ask: bool = False) -> list[str]:
        """Problems with calling this tool with ``values`` (empty list = valid call).

        With ``allow_ask``, a required argument may be ``"?"``: the user must be asked for it.
        """
        errors = [
            f"unknown argument {k!r}" for k in values if k not in {arg.name for arg in self.args}
        ]
        for arg in self.args:
            if arg.name not in values:
                if arg.required:
                    errors.append(f"missing required argument {arg.name!r}")
            elif allow_ask and arg.required and values[arg.name] == ASK_VALUE:
                continue
            elif (error := arg.check(values[arg.name])) is not None:
                errors.append(error)
        return errors


GENERATION_KEYS = ("templates", "values", "overlaps")  # task-data generation only (M6)


def load_tools(path: str | Path) -> list[Tool]:
    """The tools in one file, without the keys that only data generation uses."""
    data = json.loads(Path(path).read_text())
    return [
        Tool.from_dict({k: v for k, v in t.items() if k not in GENERATION_KEYS}, data["domain"])
        for t in data["tools"]
    ]


def load_library(directory: str | Path = defaults.TOOLS_DIR) -> dict[str, Tool]:
    """All tools in ``directory/*.json`` by name. Tool names are unique across files."""
    library: dict[str, Tool] = {}
    for path in sorted(Path(directory).glob("*.json")):
        for tool in load_tools(path):
            if tool.name in library:
                raise SchemaError(f"tool {tool.name} is defined twice ({path})")
            library[tool.name] = tool
    return library
