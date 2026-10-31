"""Golden runs: three tiny CPU runs whose logged numbers must not change.

They pin what a refactor may not move: per-step losses, learning rates, gradient norms, val
metrics, eval scores and samples, and the final weights. Regenerate the files only for an
intended change in behavior: ``EVE_GOLDEN=write uv run pytest tests/test_golden.py``.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_finetune import write_task

from eve.config import preset
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve
from eve.tokenizer.bpe import BYTE0
from eve.train.config import TrainConfig
from eve.train.finetune import finetune
from eve.train.pretrain import pretrain

GOLDEN = Path(__file__).parent / "golden"
TIMING = {"tok_s", "s_per_step"}  # depend on the machine, not the code
TOLERANCE = 1e-6


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{prefix}/{key}", out)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _flatten(item, f"{prefix}/{i}", out)
    else:
        out[prefix] = value


def numbers(run: Path) -> dict[str, Any]:
    """Every logged value of a run by ``kind/step/key``, timing left out, plus a checksum per
    weight tensor of its final model."""
    out: dict[str, Any] = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        kind, step = record.pop("kind"), record.pop("step", "-")
        if kind == "start":
            continue  # configs, not results: may gain keys
        for key in TIMING:
            record.pop(key, None)
        _flatten(record, f"{kind}/{step}", out)
    state = load_checkpoint(run / "model.pt").model.state_dict()
    for name, tensor in state.items():
        out[f"weights/{name}"] = float(tensor.double().abs().sum())
    return out


def check(name: str, run: Path) -> None:
    got, path = numbers(run), GOLDEN / f"{name}.json"
    if os.environ.get("EVE_GOLDEN") == "write":
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        return
    want = json.loads(path.read_text())
    assert sorted(got) == sorted(want)
    for key, value in want.items():
        if isinstance(value, float):
            assert got[key] == pytest.approx(value, rel=TOLERANCE, abs=TOLERANCE), key
        else:
            assert got[key] == value, key


def test_golden_pretrain(tokenizer, tmp_path):
    rng = np.random.default_rng(0)
    for split, n in [("train", 3000), ("val", 1000)]:
        rng.integers(BYTE0, 600, n).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    tokenizer.save(tmp_path / "tok.json")
    cfg = preset("tiny", max_encoder_len=128, dropout=0.1)  # dropout on: the RNG matters
    train_cfg = TrainConfig(
        steps=6, batch_size=2, warmup_steps=2, eval_every=3, eval_batches=2, log_every=1
    )
    pretrain(cfg, train_cfg, tmp_path, tmp_path / "tok.json", tmp_path / "run", "cpu")
    check("pretrain", tmp_path / "run")


def test_golden_finetune(library, tokenizer, tmp_path):
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "pre.pt", Eve(preset("tiny")), tokenizer, {"step": 1})
    write_task(tmp_path / "task", library)
    train_cfg = TrainConfig(steps=6, batch_size=3, warmup_steps=2, eval_every=3, log_every=1)
    finetune(
        train_cfg, tmp_path / "task", tmp_path / "run", "cpu", library,
        init=tmp_path / "pre.pt", overrides={"dropout": 0.1}, evals_dir=tmp_path / "task",
    )  # fmt: skip
    check("finetune-joint", tmp_path / "run")
