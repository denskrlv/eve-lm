"""A shard of generated fine-tuning data: its tools, its requests, and where it stands.

A shard is a directory ``NNNN/`` under the shards root (``data/datagen/shards``) holding
``topics.json`` (the topics it was given), ``tools.json``, ``requests.jsonl``, ``review.json``,
``report.json``, ``stats.json`` and ``status``. Every file is written whole (``write_atomic``),
so a stopped run loses at most the step in flight. A ``.lock`` file holding a PID marks the
shard as taken by a running process.
"""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eve.data.render import Call, TaskExample
from eve.data.schema import Arg, SchemaError, Tool

TOOLS_PER_SHARD = 25
TOPICS_PER_SHARD = 5
STATUSES = ("new", "tools", "requests", "reviewed", "approved", "flagged")
DONE = ("approved", "flagged")
LOCK = ".lock"


@dataclass(frozen=True)
class GenTool:
    """An invented tool: several names and descriptions for one capability."""

    id: str
    topic: str  # "area > subtopic"
    names: tuple[str, ...]  # the first is canonical: calls in requests.jsonl use it
    descriptions: tuple[str, ...]
    args: tuple[Arg, ...] = ()
    capability: str = ""  # "" = the canonical name; a library twin's capability otherwise

    def __post_init__(self) -> None:
        if not self.names or not self.descriptions:
            raise SchemaError(f"tool {self.id}: needs at least one name and one description")
        if not self.capability:
            object.__setattr__(self, "capability", self.names[0])
        for i in range(len(self.names)):
            self.tool(i, 0)  # every name builds a valid Tool
        for j in range(len(self.descriptions)):
            self.tool(0, j)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenTool":
        """A tool from its JSON form in ``tools.json``."""
        try:
            return cls(
                str(data["id"]),
                str(data["topic"]),
                tuple(data["names"]),
                tuple(data["descriptions"]),
                tuple(Arg.from_dict(a) for a in data.get("args", ())),
                str(data.get("capability", "")),
            )
        except (KeyError, TypeError) as error:
            raise SchemaError(f"tool {data.get('id')!r}: {error!r}") from None

    def to_dict(self) -> dict[str, Any]:
        """The JSON form."""
        return {
            "id": self.id,
            "topic": self.topic,
            "names": list(self.names),
            "descriptions": list(self.descriptions),
            "args": [a.to_dict() for a in self.args],
            "capability": self.capability,
        }

    @property
    def area(self) -> str:
        """The topic's area: "garden" for "garden > pests"."""
        return self.topic.split(" > ")[0].strip()

    def tool(self, name: int = 0, description: int = 0) -> Tool:
        """One variant as a schema the model reads."""
        return Tool(self.names[name], self.descriptions[description], self.args, self.area)

    @property
    def texts(self) -> tuple[str, ...]:
        """Every name (as words) and description: what a request must share no word with."""
        return tuple(n.replace("_", " ") for n in self.names) + self.descriptions


@dataclass(frozen=True)
class Row:
    """One generated request with its label, for one tool of the shard."""

    id: str
    tool: str  # a GenTool id
    utterance: str
    intent: str  # call, ask or unsupported
    args: dict[str, Any] | None  # None for unsupported
    card: dict[str, str] = field(default_factory=dict, compare=False)  # what it was asked to be
    writer: str = ""  # the model that wrote it, or "fixed" after Claude's review

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Row":
        """A row from its line in ``requests.jsonl``."""
        calls = data.get("calls") or []
        args = dict(calls[0]["args"]) if calls else None
        return cls(
            data["id"],
            data["tool"],
            data["utterance"],
            data["intent"],
            args,
            dict(data.get("card", {})),
            data.get("writer", ""),
        )

    def to_dict(self, name: str = "") -> dict[str, Any]:
        """The line in ``requests.jsonl``; ``name`` is the tool's canonical name for the call."""
        calls = [] if self.args is None else [{"tool": name, "args": self.args}]
        return {
            "id": self.id,
            "tool": self.tool,
            "utterance": self.utterance,
            "intent": self.intent,
            "calls": calls,
            "card": self.card,
            "writer": self.writer,
        }

    def call(self, tool: Tool) -> tuple[Call, ...]:
        """The row's call to ``tool`` (any variant of its GenTool), or none."""
        return () if self.args is None else (Call(tool.name, dict(self.args)),)

    def example(self, tool: GenTool, name: int = 0, description: int = 0) -> TaskExample:
        """The row with only its own tool in the prompt. Raises ``SchemaError``."""
        shown = tool.tool(name, description)
        return TaskExample(self.utterance, (shown,), self.intent, self.call(shown))


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` whole: a temp file in the same directory, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Shard:
    """One shard directory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.number = int(self.path.name)

    def __repr__(self) -> str:
        return f"Shard({self.path.name})"

    def read_json(self, name: str, default: Any = None) -> Any:
        """A JSON file of the shard, or ``default`` if it doesn't exist."""
        path = self.path / name
        return json.loads(path.read_text()) if path.exists() else default

    def write_json(self, name: str, data: Any) -> None:
        """Write a JSON file of the shard whole."""
        write_atomic(self.path / name, json.dumps(data, indent=1, ensure_ascii=False) + "\n")

    def status(self) -> str:
        """Where the shard stands: one of ``STATUSES``."""
        path = self.path / "status"
        return path.read_text().strip() if path.exists() else "new"

    def set_status(self, status: str) -> None:
        """Move the shard to ``status``."""
        if status not in STATUSES:
            raise ValueError(f"unknown shard status {status!r}; one of {STATUSES}")
        write_atomic(self.path / "status", status + "\n")

    def topics(self) -> list[str]:
        """The topics the shard was given."""
        return list(self.read_json("topics.json", []))

    def tools(self) -> list[GenTool]:
        """The shard's tools."""
        return [GenTool.from_dict(t) for t in self.read_json("tools.json", [])]

    def save_tools(self, tools: Sequence[GenTool]) -> None:
        """Replace the shard's tools."""
        self.write_json("tools.json", [t.to_dict() for t in tools])

    def rows(self) -> list[Row]:
        """The shard's requests."""
        path = self.path / "requests.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        return [Row.from_dict(json.loads(line)) for line in lines if line.strip()]

    def save_rows(self, rows: Sequence[Row]) -> None:
        """Replace the shard's requests."""
        names = {t.id: t.names[0] for t in self.tools()}
        lines = [json.dumps(r.to_dict(names.get(r.tool, "")), ensure_ascii=False) for r in rows]
        write_atomic(self.path / "requests.jsonl", "".join(line + "\n" for line in lines))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process
    return True


def claim(shard: Shard) -> bool:
    """Take the shard for this process. False if a live process holds it; a lock left by a
    dead process is taken over."""
    path = shard.path / LOCK
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _alive(pid):
                return False
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return True
    return False


def release(shard: Shard) -> None:
    """Give the shard back."""
    (shard.path / LOCK).unlink(missing_ok=True)


def topic_slice(topics: Sequence[str], number: int) -> list[str]:
    """The topics of shard ``number``: a fixed slice of the list, so two processes making
    shards at once never give two shards the same topics."""
    start = (number - 1) * TOPICS_PER_SHARD
    picked = list(topics[start : start + TOPICS_PER_SHARD])
    if len(picked) < TOPICS_PER_SHARD:
        raise ValueError(
            f"topics.txt has no unused topics left for shard {number:04d} "
            f"({len(topics)} topics, {TOPICS_PER_SHARD} per shard): add more lines"
        )
    return picked


def shards(root: Path) -> list[Shard]:
    """Every shard under ``root``, by number."""
    root = Path(root)
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted((Shard(p) for p in dirs), key=lambda s: s.number)


def new_shard(root: Path, topics: Sequence[str]) -> Shard:
    """A new shard numbered after the highest one under ``root``, with its topics. Raises
    ``ValueError`` when the topics have run out."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    number = max((s.number for s in shards(root)), default=0) + 1
    while True:
        picked = topic_slice(topics, number)
        try:
            (root / f"{number:04d}").mkdir()
        except FileExistsError:
            number += 1
            continue
        shard = Shard(root / f"{number:04d}")
        shard.write_json("topics.json", picked)
        shard.set_status("new")
        return shard
