import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_lines(path: str | Path) -> list[str]:
    """Non-empty lines that aren't # comments."""
    lines = (line.strip() for line in Path(path).read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """One dict per non-empty line."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """One JSON line per row, creating the parent directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
