from colibri.data.jsonl import read_lines


def test_read_lines_skips_blank_lines_and_comments_even_indented(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("# header\n  turn on the fan\n\n   # indented comment\nlock the door  \n")
    assert read_lines(path) == ["turn on the fan", "lock the door"]
