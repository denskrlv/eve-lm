import json
from pathlib import Path
from typing import Any

import torch

from eve.config import preset
from eve.model.splitter import Splitter, SplitterConfig
from eve.tasks.split import split_data, split_loss, split_metrics
from eve.tokenizer.bpe import BPETokenizer

CFG = SplitterConfig(**preset("tiny").to_dict())


def _row(text: str, actions: list[list[list[int]]] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"utterance": text, "source": "template", "calls": [{}]}
    if actions is not None:
        row |= {"calls": [{}] * len(actions), "actions": actions}
    return row


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    path.mkdir()
    (path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _multi(i: int) -> dict[str, Any]:
    text = f"turn off fan {i} and lock door {i}"
    cut = text.index(" and")
    return _row(text, [[[0, cut]], [[cut + 5, len(text)]]])


def test_collate_shapes_fixed(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    rows = [_row("lights off"), _row(" ".join(["lights"] * 30)), _multi(0), _multi(1)]
    data = split_data(tokenizer, CFG, 0, path=_write(tmp_path / "t", rows), val_share=0.0)
    items = sorted(data.train, key=lambda item: len(item.ids))
    batch = data.collate([items[0], items[-1]])
    assert batch["enc_ids"].shape == (2, 64) and batch["members"].shape == (2, 8, 64)
    assert batch["exists"].shape == (2, 8) and batch["word_mask"].shape == (2, 64)
    assert batch["word_mask"].sum(1).tolist() == [len(items[0].first), len(items[-1].first)]


def test_single_share_cap(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    rows = [_row(f"lights off number {i}") for i in range(90)] + [_multi(i) for i in range(10)]
    data = split_data(tokenizer, CFG, 0, path=_write(tmp_path / "t", rows), single_share=0.5)
    items = list(data.train) + list(data.val)
    assert sum(len(item.members) == 1 for item in items) == 10
    assert sum(len(item.members) == 2 for item in items) == 10


def test_val_grouped_by_text(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    rows = [_multi(i % 5) for i in range(40)]  # 5 texts, 8 copies each
    data = split_data(tokenizer, CFG, 0, path=_write(tmp_path / "t", rows), val_share=0.3)
    train, val = {item.text for item in data.train}, {item.text for item in data.val}
    assert train and val and not train & val


def test_rows_without_known_actions_are_skipped(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    rows = [_multi(0), {"utterance": "a and b", "source": "chat", "calls": [{}, {}]}]
    data = split_data(tokenizer, CFG, 0, path=_write(tmp_path / "t", rows), val_share=0.0)
    assert [item.text for item in data.train] == [rows[0]["utterance"]]


def test_loss_and_metrics_on_tiny(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    rows = [_row("lights off"), _multi(0), _multi(1)]
    data = split_data(tokenizer, CFG, 0, path=_write(tmp_path / "t", rows), val_share=0.0)
    torch.manual_seed(0)
    model = Splitter(CFG)
    batch = data.collate(list(data.train))
    losses = split_loss(model, batch)
    assert set(losses) == {"exists", "members"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    metrics = split_metrics(model, batch)
    assert {"exact_split", "wrong_split"} <= set(metrics)
    assert all(0 <= metrics[k].item() <= 1 for k in ("exact_split", "wrong_split"))
