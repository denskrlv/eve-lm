import os
import sys
from typing import Any, TextIO

from tqdm import tqdm

BAR_FORMAT = (
    "{desc} {percentage:3.0f}% │{bar:30}│ {n_fmt}/{total_fmt} {unit} "
    "· {elapsed} < {remaining} {stats}"
)


class Bar(tqdm):  # type: ignore[type-arg]  # tqdm is generic only in its stubs
    """tqdm with a ``stats`` field: tqdm's own ``postfix`` always starts with ", "."""

    stats = ""

    @property
    def format_dict(self) -> dict[str, Any]:
        return {**super().format_dict, "stats": self.stats}

    def set_stats(self, *parts: str) -> None:
        self.stats = "· " + " · ".join(parts) if parts else ""
        self.refresh()


def color(stream: TextIO = sys.stderr) -> bool:
    """ANSI colors only in a real terminal, and never with NO_COLOR set (no-color.org)."""
    return stream.isatty() and "NO_COLOR" not in os.environ


def style(text: str, *codes: str, stream: TextIO = sys.stderr) -> str:
    """``style("done", "1", "32")`` -> bold green "done" in a terminal, plain text otherwise."""
    return f"\033[{';'.join(codes)}m{text}\033[0m" if codes and color(stream) else text


def marks(stream: TextIO = sys.stdout) -> tuple[str, str]:
    """How filled-in spans are marked: [brackets], bold cyan in a terminal (an empty fill
    still shows as ``[]``)."""
    return ("\033[1;36m[", "]\033[0m") if color(stream) else ("[", "]")


def bar(total: int, desc: str, unit: str, initial: int = 0, leave: bool = True) -> Bar:
    """A progress bar on stderr. It switches itself off when stderr isn't a terminal (a log
    file, pytest), so callers check ``bar.disable`` and print plain lines instead."""
    return Bar(
        total=total,
        initial=initial,
        desc=style(desc, "1"),
        unit=unit,
        bar_format=BAR_FORMAT,
        dynamic_ncols=True,
        leave=leave,
        disable=None,
        colour="cyan",
    )
