"""The tools a developer registers with Eve once, instead of passing them to every request.

Eve reads at most ``max_tools`` (5) tools per request. With more registered, ``search`` picks the
ones to show it: BM25 over each tool's name, description, argument names, options and argument
descriptions. Word overlap misses paraphrases ("20 plus 10" shares no word with "Add two
numbers"), so a retriever trained on meaning will replace it (M11); the interface stays.

A registry is saved as JSON schemas. Functions are code and can't be saved: after ``load``, add
them again and each one takes the place of the schema with its name.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from eve.api.errors import ToolError
from eve.api.tools import ToolLike, to_tool
from eve.data.schema import Tool

K1, B = 1.2, 0.75  # BM25's usual term-saturation and length-normalization constants
_WORD = re.compile(r"[a-z0-9]+")  # snake_case names split at "_"


def words(text: str) -> list[str]:
    """Lowercase words, plurals folded: "lights" and "light" match."""
    out = []
    for word in _WORD.findall(text.lower()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


def _document(tool: Tool) -> list[str]:
    """What a request is matched against: the name counts twice, being the most telling."""
    parts = [tool.name, tool.name, tool.description]
    for arg in tool.args:
        parts += [arg.name, arg.description, *arg.options]
    return words(" ".join(parts))


class ToolBox:
    """Registered tools by name, with their functions if they are functions."""

    def __init__(self, tools: Iterable[ToolLike] = ()) -> None:
        self._tools: dict[str, tuple[Tool, Callable[..., Any] | None]] = {}
        self._docs: dict[str, Counter[str]] = {}
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
            self._docs[tool.name] = Counter(_document(tool))

    def remove(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(f"no tool called {name!r}")
        del self._tools[name], self._docs[name]

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
        return list(self._tools)

    @property
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Name -> function, for the tools registered as functions."""
        return {name: fn for name, (_, fn) in self._tools.items() if fn is not None}

    def scores(self, text: str) -> dict[str, float]:
        """The BM25 score of every tool for ``text``, in registration order."""
        query = set(words(text))
        n = len(self._docs)
        if not n:
            return {}
        mean_length = sum(sum(d.values()) for d in self._docs.values()) / n
        df = Counter(word for doc in self._docs.values() for word in doc.keys() & query)
        out = {}
        for name, doc in self._docs.items():
            length, score = sum(doc.values()), 0.0
            for word in query & doc.keys():
                idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                tf = doc[word]
                score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / mean_length))
            out[name] = score
        return out

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
        return cls(data["tools"] if isinstance(data, dict) else data)

    def __repr__(self) -> str:
        return f"ToolBox({len(self)} tools)"
