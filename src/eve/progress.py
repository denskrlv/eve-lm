from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from rich import box
from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table as Grid
from rich.text import Text
from rich.tree import Tree

# Writes to whatever sys.stdout is at the time, like print (pytest and redirects swap it).
# Colors and the live bar only in a terminal; NO_COLOR (no-color.org) turns colors off.
console = Console(highlight=False)

BAR_WIDTH = 30


class Bar:
    """A progress bar with a ``stats`` field after it, e.g. the loss and speed.

    It is off when the console isn't a terminal (a log file, pytest), so callers check
    ``disable`` and print plain lines instead. While it runs, ``write`` and anything printed
    to stdout (log lines) appear above it.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str,
        initial: int = 0,
        leave: bool = True,
        disable: bool | None = None,
    ) -> None:
        self.disable = not console.is_terminal if disable is None else disable
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(BAR_WIDTH, style="grey23", complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TextColumn(f"{{task.completed:,.0f}}/{{task.total:,.0f}} {unit}", style="dim"),
            TimeElapsedColumn(),
            TextColumn("<", style="dim"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[stats]}", markup=False),  # file names may hold [ ]
            console=console,
            transient=not leave,
            disable=self.disable,
        )
        self._task = self._progress.add_task(desc, total=total, completed=initial, stats="")
        self._progress.start()

    def update(self, n: int = 1) -> None:
        """Advance by ``n``."""
        self._progress.advance(self._task, n)

    def set_stats(self, *parts: str) -> None:
        """Show ``parts`` after the bar, joined by " · "."""
        self._progress.update(self._task, stats="· " + " · ".join(parts) if parts else "")

    def write(self, renderable: RenderableType) -> None:
        """Print above the bar; a ``str`` as plain text, never as rich markup."""
        console.print(Text(renderable) if isinstance(renderable, str) else renderable)

    def close(self) -> None:
        """Stop the bar, leaving its last state on screen if ``leave``."""
        self._progress.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


def bar(total: int, desc: str, unit: str, initial: int = 0, leave: bool = True) -> Bar:
    """A progress bar for ``total`` ``unit``s, labelled ``desc``; see ``Bar``."""
    return Bar(total, desc, unit, initial=initial, leave=leave)


def format_metric(name: str, value: float) -> str:
    """Accuracies, rates and recalls as percentages (``97.5%``, also the retrieval scores
    ``all@5`` and ``recall@5``), losses with four decimals."""
    percent = name.endswith(("acc", "rate", "recall")) or "@" in name
    return f"{value:.1%}" if percent else f"{value:.4f}"


def higher_is_better(name: str) -> bool:
    """Accuracies and recalls should go up; losses and rates (e.g. ``call_stop_rate``) down."""
    return name.endswith(("acc", "recall")) or "@" in name


def format_duration(seconds: float) -> str:
    """``42s``, ``3m07s`` or ``2h05m``."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def rows(items: Sequence[tuple[str, str | Text]], indent: int = 0) -> RenderableType:
    """Labelled lines with the values lined up and the labels dimmed, ``indent`` spaces in. An
    empty label continues the row above (``samples`` followed by more sample lines). A ``str``
    value is plain text."""
    grid = Grid.grid(padding=(0, 3, 0, 0))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    for label, value in items:
        grid.add_row(label, Text(value, overflow="fold") if isinstance(value, str) else value)
    return Padding(grid, (0, 0, 0, indent)) if indent else grid


def panel(title: str, items: Sequence[tuple[str, str | Text]]) -> Panel:
    """Labelled ``rows`` in a box with a bold title: a command's banner."""
    return Panel(
        rows(items),
        title=Text(title, style="bold"),
        title_align="left",
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    )


def table(*labels: str, first_style: str = "") -> Grid:
    """A table in the CLI's style: bold labels over a rule, the first column (row names) on
    the left in ``first_style``, the others (numbers) on the right."""
    out = Grid(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    out.add_column(labels[0], style=first_style, no_wrap=True)
    for label in labels[1:]:
        out.add_column(label, justify="right", no_wrap=True)
    return out


def success(text: str, path: object = None) -> None:
    """``✓ text → path``: something finished, and where its result is."""
    line = Text.assemble(("✓ ", "bold green"), text)
    if path is not None:
        line.append_text(Text.assemble(" → ", (str(path), "cyan")))
    console.print(line, soft_wrap=True)  # a long path wraps like plain text


def warn(text: str) -> None:
    """``! text`` in yellow: something the user should know, but not an error."""
    console.print(Text(f"! {text}", style="yellow"), soft_wrap=True)


def _branch(node: Tree, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _branch(node.add(Text(str(key), style="bold")), value)
        else:
            shown = f"{value:,}" if isinstance(value, int) else str(value)
            node.add(Text.assemble((str(key), "dim"), "  ", shown))


def tree(title: str, data: dict[str, Any]) -> Tree:
    """Nested dicts as a tree under a bold ``title``: a branch per dict, a leaf per value."""
    root = Tree(Text(title, style="bold cyan"), guide_style="dim")
    _branch(root, data)
    return root


@dataclass(frozen=True)
class Table:
    """Right-aligned columns printed a row at a time, e.g. one row per eval of a run.

    Rows are printed one by one rather than as one rich table, so each eval's row appears as
    soon as it exists and the table can grow longer than the screen.
    """

    labels: Sequence[str]
    min_width: int = 7

    def _widths(self) -> list[int]:
        return [max(len(label), self.min_width) for label in self.labels]

    def _line(self, cells: Sequence[Text]) -> Text:
        line = Text("  ")
        for i, (cell, width) in enumerate(zip(cells, self._widths(), strict=True)):
            line.append(" " * (width - cell.cell_len + (2 if i else 0)))
            line.append_text(cell)
        return line

    def header(self) -> Text:
        """The labels in bold over a dim rule."""
        rule = "─" * (sum(self._widths()) + 2 * (len(self.labels) - 1))
        line = self._line([Text(label, style="bold") for label in self.labels])
        return Text.assemble(line, "\n", (f"  {rule}", "dim"))

    def row(self, cells: Sequence[str | Text]) -> Text:
        """One row; ``cells`` are already formatted, one per label (a ``Text`` keeps its style)."""
        return self._line([Text(c) if isinstance(c, str) else c for c in cells])
