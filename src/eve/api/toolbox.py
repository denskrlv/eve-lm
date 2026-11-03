"""The tools a developer registers with Eve once, instead of passing them to every request.

Eve reads at most ``max_tools`` (5) tools per request. With more registered, ``search`` picks the
ones to show it, ranked by the box's ``scorer``: ``BM25`` (the default) or ``Dense`` (a trained
retriever), from ``eve.infer.search``.

A registry is saved as JSON schemas. Functions are code and can't be saved: after ``load``, add
them again and each one takes the place of the schema with its name.
"""

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from eve.api.errors import ToolError
from eve.api.tools import ToolLike, to_tool
from eve.data.schema import Tool
from eve.infer.search import BM25, Dense, Scorer, words

__all__ = ["BM25", "Dense", "Scorer", "ToolBox", "words"]  # the scorers' old home re-exports them


class ToolBox:
    """Registered tools by name, with their functions if they are functions."""

    def __init__(self, tools: Iterable[ToolLike] = (), scorer: Scorer | None = None) -> None:
        self._tools: dict[str, tuple[Tool, Callable[..., Any] | None]] = {}
        self.scorer: Scorer = BM25() if scorer is None else scorer  # what ``search`` ranks by
        self.add(*tools)

    def add(self, *tools: ToolLike | Iterable[ToolLike]) -> None:
        """Register tools: functions, JSON schema dicts or ``Tool`` objects, or lists of them.
        A tool with the name of a registered one takes its place."""
        for item in tools:
            if isinstance(item, list | tuple | ToolBox):
                self.add(*item)
                continue
            tool, function = to_tool(item)  # type: ignore[arg-type]
            self._tools[tool.name] = (tool, function)

    def remove(self, name: str) -> None:
        """Unregister the tool called ``name``; ``KeyError`` if there is none."""
        if name not in self._tools:
            raise KeyError(f"no tool called {name!r}")
        del self._tools[name]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return (tool for tool, _ in self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name][0]

    @property
    def names(self) -> list[str]:
        """The registered tools' names, in registration order."""
        return list(self._tools)

    @property
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Name -> function, for the tools registered as functions."""
        return {name: fn for name, (_, fn) in self._tools.items() if fn is not None}

    def scores(self, text: str) -> dict[str, float]:
        """The ``scorer``'s score of every tool for ``text``, in registration order."""
        tools = list(self)
        return dict(zip(self.names, self.scorer.scores(text, tools), strict=True))

    def search(self, text: str, k: int) -> list[Tool]:
        """The ``k`` tools that best match ``text``, best first; ties keep registration order."""
        scores = self.scores(text)
        ranked = sorted(scores, key=lambda name: -scores[name])  # sorted() is stable
        return [self._tools[name][0] for name in ranked[:k]]

    def save(self, path: str | Path) -> None:
        """The schemas as JSON (``{"tools": [...]}``, Eve's format); functions aren't saved."""
        data = {"tools": [tool.to_dict() for tool in self]}
        Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ToolBox":
        """Schemas saved by ``save``, or any JSON list of tool schemas (Eve, OpenAI, MCP)."""
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ToolError(f"can't read tools from {path}: {error}") from None
        if isinstance(data, dict):
            if "tools" not in data:
                raise ToolError(f'can\'t read tools from {path}: no "tools" list')
            data = data["tools"]
        elif not isinstance(data, list):
            raise ToolError(f"can't read tools from {path}: give a list of tool schemas")
        return cls(data)

    def __repr__(self) -> str:
        return f"ToolBox({len(self)} tools)"
