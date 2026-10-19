import json
from dataclasses import dataclass, field
from pathlib import Path

from colibri.data.generate.templates import check_template
from colibri.data.schema import GENERATION_KEYS, SchemaError, Tool


@dataclass(frozen=True)
class ToolSpec:
    """A tool plus what data generation needs to write requests for it.

    ``capability`` names what the tool can do. Two tools with the same capability never appear
    in one prompt. ``overlaps`` lists tools whose requests this tool could also serve ("*" = any
    request, e.g. a web search). Such a tool is never offered for a request labelled UNSUPPORTED.
    ``weight`` is how often the tool is picked as the correct one, relative to others. A tool
    without templates can only be a distractor (or the gold tool of an external example).
    """

    tool: Tool
    templates: tuple[str, ...]
    values: dict[str, tuple[str, ...]] = field(default_factory=dict, compare=False)
    capability: str = ""
    overlaps: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        for template in self.templates:
            if errors := check_template(self.tool, template, self.values):
                raise SchemaError(f"tool {self.tool.name}, {template!r}: {'; '.join(errors)}")
        if not self.capability:
            object.__setattr__(self, "capability", self.tool.name)

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def domain(self) -> str:
        return self.tool.domain


def load_specs(directory: str | Path) -> list[ToolSpec]:
    """Every tool in ``directory/*.json`` with its templates and value pools."""
    specs = []
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text())
        for raw_tool in data["tools"]:
            tool = Tool.from_dict(
                {k: v for k, v in raw_tool.items() if k not in GENERATION_KEYS}, data["domain"]
            )
            specs.append(
                ToolSpec(
                    tool,
                    tuple(raw_tool.get("templates", ())),
                    {k: tuple(v) for k, v in raw_tool.get("values", {}).items()},
                    overlaps=tuple(raw_tool.get("overlaps", ())),
                )
            )
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise SchemaError(f"duplicate tool names in {directory}")
    return specs
