import io

from rich.console import Console, RenderableType

from eve.progress import (
    Table,
    format_duration,
    format_metric,
    panel,
    rows,
    success,
    table,
    tree,
    warn,
)


def render(renderable: RenderableType) -> list[str]:
    out = io.StringIO()
    Console(file=out, width=60, color_system=None).print(renderable)
    return out.getvalue().rstrip("\n").split("\n")


def test_durations_are_short():
    assert [format_duration(s) for s in (42, 187, 7500)] == ["42s", "3m07s", "2h05m"]


def test_recalls_are_percentages():
    assert format_metric("macro_recall", 0.523) == "52.3%"
    assert format_metric("val_lm", 3.2) == "3.2000"


def test_rows_line_up_values_and_keep_brackets():
    lines = render(rows([("model", "tiny"), ("schedule", "40 steps"), ("", "[Paris]")]))
    assert [line.rstrip() for line in lines] == [
        "model      tiny",
        "schedule   40 steps",
        "           [Paris]",
    ]


def test_panel_boxes_its_rows_under_the_title():
    lines = render(panel("eve pretrain", [("device", "cpu")]))
    assert "eve pretrain" in lines[0]
    assert any("device   cpu" in line for line in lines)


def test_table_rows_line_up_with_the_header():
    table = Table(["step", "held-out macro"])
    header, rule = table.header().plain.split("\n")
    row = table.row(["1,000", "52.0%"]).plain
    assert len(header) == len(rule) == len(row)
    assert row.endswith("52.0%") and header.endswith("held-out macro")


def test_table_puts_names_left_and_numbers_right():
    out = table("", "words")
    out.add_row("commands", "285")
    out.add_row("web text", "79,063")
    header, rule, first, second = [line for line in render(out) if line.strip()]
    assert first.index("commands") < second.index("79,063")
    assert first.rstrip().endswith("285") and second.rstrip().endswith("79,063")


def test_tree_shows_nested_counts_with_separators():
    lines = render(tree("external data", {"xlam": {"kept": 34376, "reasons": {"ok": 2}}}))
    assert lines[0] == "external data"
    assert any("kept  34,376" in line for line in lines)
    assert any("reasons" in line for line in lines) and any("ok  2" in line for line in lines)


def test_status_lines(capsys):
    success("built", "data/task/report.json")
    warn("2 lines rejected")
    assert capsys.readouterr().out.splitlines() == [
        "✓ built → data/task/report.json",
        "! 2 lines rejected",
    ]
