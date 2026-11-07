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


def _run(tmp_path: Path, tokenizer) -> Path:  # type: ignore[no-untyped-def]
    import torch

    from eve.config import preset
    from eve.model.checkpoint import save_checkpoint
    from eve.model.splitter import Splitter, SplitterConfig

    torch.manual_seed(0)
    run = tmp_path / "run"
    model = Splitter(SplitterConfig(**preset("tiny").to_dict()))
    save_checkpoint(run / "model.pt", model, tokenizer, {"task": "split"})
    task = tmp_path / "task"
    task.mkdir()
    rows = []
    for i in range(20):
        text = f"turn off fan {i} and lock door {i}"
        cut = text.index(" and")
        rows.append(
            {
                "utterance": text,
                "source": "template",
                "calls": [{}, {}],
                "actions": [[[0, cut]], [[cut + 5, len(text)]]],
            }
        )
        rows.append({"utterance": f"lights off number {i}", "source": "massive", "calls": [{}]})
    (task / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = f'[data]\npath = "{task}"\nsingle_share = 0.5\nval_share = 0.5\n\n[train]\nseed = 0\n'
    (run / "config.toml").write_text(config)
    return run


def test_split_tune_writes_thresholds(tmp_path: Path, tokenizer, capsys) -> None:  # type: ignore[no-untyped-def]
    run = _run(tmp_path, tokenizer)
    main(["split", "tune", str(run), "--device", "cpu"])
    th = json.loads((run / "thresholds.json").read_text())
    assert set(th) == {"exists", "member"}
    assert "exists" in capsys.readouterr().out


def test_split_evaluate_and_ask(tmp_path: Path, tokenizer, library, capsys) -> None:  # type: ignore[no-untyped-def]
    from helpers import save_joint, write_task

    run = _run(tmp_path, tokenizer)
    with pytest.raises(SystemExit, match="eve split tune"):
        main(["split", "evaluate", str(run), "--device", "cpu"])
    (run / "thresholds.json").write_text(json.dumps({"exists": 0.5, "member": 0.5}))
    eve = save_joint(tmp_path / "eve.pt", tokenizer)
    write_task(tmp_path / "evals", library)
    hard = tmp_path / "hard.jsonl"
    hard.write_text(
        json.dumps({"utterance": "lights off", "actions": ["lights off"], "kind": "no-split"})
        + "\n"
    )
    argv = ["split", "evaluate", str(run), "--model", str(eve), "--evals", str(tmp_path / "evals")]
    main([*argv, "--sets", "val", "--hard", str(hard), "--device", "cpu", "--json"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(out["sets"]["val"]["pipeline"]) == {"alone", "split"}
    assert "exact_split" in out["sets"]["val"]["splitter"]
    assert out["hard"]["no-split"] in (0.0, 1.0)

    main(["split", "ask", str(run), "turn off the fan and lock the door", "--device", "cpu"])
    assert "turn off the fan" in capsys.readouterr().out


def test_split_tune_takes_a_checkpoint_file(tmp_path: Path, tokenizer) -> None:  # type: ignore[no-untyped-def]
    run = _run(tmp_path, tokenizer)
    (run / "model.pt").rename(run / "last.pt")
    main(["split", "tune", str(run / "last.pt"), "--device", "cpu"])
    assert (run / "thresholds.json").is_file()
