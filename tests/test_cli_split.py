import json
from pathlib import Path

import pytest

from eve.cli import main


def test_split_labels_prints_actions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = "turn off the fan and lock the door"
    rows = [
        {
            "utterance": text,
            "source": "xlam",
            "calls": [{}, {}],
            "actions": [[[0, 16]], [[21, 34]]],
        },
        {"utterance": "lights off", "source": "xlam", "calls": [{}]},
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    main(["split", "labels", "--path", str(path), "--source", "xlam", "--sample", "5"])
    out = capsys.readouterr().out
    assert text in out and "1. turn off the fan" in out and "2. lock the door" in out
    assert "lights off" not in out  # only multi-action rows are shown
