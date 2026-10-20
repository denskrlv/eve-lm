import json
from pathlib import Path
from typing import Any

import pytest

from eve.data.generate.build import (
    Sources,
    build,
    check_written,
    make_batch,
    next_batch_name,
)
from eve.data.jsonl import read_jsonl

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def sources():
    return Sources.load(ROOT / "data/tools", ROOT / "data/generation")


def rephrased(scenario: dict[str, Any]) -> str:
    """The scenario's meaning, changed just enough not to be a copy; small talk for NONE."""
    return f"so {scenario['meaning']}" if scenario["meaning"] else "hello there"


def test_batch_and_check(sources):
    batch = make_batch(sources, "batch-0001", 20, "val", seed=1)
    assert batch == make_batch(sources, "batch-0001", 20, "val", seed=1)
    assert len(batch["scenarios"]) == 20
    assert all(
        isinstance(t, dict) for s in batch["scenarios"] for t in s["tools"]
    )  # self-contained
    assert all(t["name"] in batch["schemas"] for s in batch["scenarios"] for t in s["tools"])

    # A rephrased template meaning always passes; a changed value doesn't.
    written = [{"id": s["id"], "utterance": rephrased(s)} for s in batch["scenarios"]]
    examples, rejects = check_written(batch, written, sources.library)
    assert len(examples) == 20 and rejects == []
    call = next(
        s
        for s in batch["scenarios"]
        if s["intent"] == "call"
        and any(isinstance(v, str) for c in s["calls"] for v in c["args"].values())
    )
    _, rejects = check_written(
        batch, [{"id": call["id"], "utterance": "do the thing"}], sources.library
    )
    assert len(rejects) == 1


def test_probe_batches_use_held_out_tools(sources):
    batch = make_batch(sources, "batch-0002", 60, "probe", seed=2)
    heldout = {s.name for s in sources.heldout}
    assert any(c["tool"] in heldout for s in batch["scenarios"] for c in s["calls"])


def test_build(tmp_path, sources):
    gen = tmp_path / "generation"
    for name in ("chitchat.txt", "unsupported.txt"):
        (gen / name).parent.mkdir(exist_ok=True)
        (gen / name).write_text((ROOT / "data/generation" / name).read_text())
    (gen / "batches").mkdir()
    (gen / "written").mkdir()
    batch = make_batch(sources, next_batch_name(gen / "batches"), 10, "val", seed=3)
    assert batch["batch"] == "batch-0001"
    (gen / "batches/batch-0001.json").write_text(json.dumps(batch))
    lines = [{"id": s["id"], "utterance": rephrased(s)} for s in batch["scenarios"]]
    lines.append({"id": "batch-0001-999", "utterance": "unknown id"})
    (gen / "written/batch-0001.jsonl").write_text("\n".join(json.dumps(x) for x in lines))

    report = build(
        tmp_path / "task",
        300,
        50,
        seed=0,
        tools_dir=ROOT / "data/tools",
        generation_dir=gen,
        external_dir=tmp_path / "no-external",
    )
    assert (
        report["splits"]["train"]["examples"]
        == 300 - report["removed_duplicates"]["train (in val)"]
    )
    assert report["splits"]["val"]["examples"] == 10
    assert report["rejected_chat_lines"] == 1
    val = {r["utterance"] for r in read_jsonl(tmp_path / "task/val.jsonl")}
    assert not val & {r["utterance"] for r in read_jsonl(tmp_path / "task/train.jsonl")}
    heldout = read_jsonl(tmp_path / "task/heldout-templates.jsonl")
    assert len(heldout) == 50


def test_batch_ask_share(sources):
    batch = make_batch(sources, "batch-0003", 100, "val", seed=4, ask=0.5)
    asks = sum(s["intent"] == "ask" for s in batch["scenarios"])
    assert 35 <= asks <= 65 and all(
        s["missing"] for s in batch["scenarios"] if s["intent"] == "ask"
    )


def test_check_rejects_a_copied_meaning(sources: Sources) -> None:
    batch = make_batch(sources, "batch-0004", 20, "val", seed=5)
    call = next(s for s in batch["scenarios"] if s["intent"] == "call")
    copied = [{"id": call["id"], "utterance": call["meaning"].upper() + "!"}]
    _, rejects = check_written(batch, copied, sources.library)
    assert [errors for _, errors in rejects] == [["the utterance copies the meaning; rephrase it"]]


def test_check_rejects_a_conversion_no_one_asks_for(sources: Sources) -> None:
    tool = sources.library["convert_currency"]
    scenario = {
        "id": "batch-0005-000",
        "tools": [tool.to_dict()],
        "intent": "call",
        "calls": [
            {
                "tool": tool.name,
                "args": {"amount": 70, "from_currency": "USD", "to_currency": "USD"},
            }
        ],
        "meaning": "how much is 70 USD in USD",
    }
    line = {"id": scenario["id"], "utterance": "what's 70 USD worth in USD"}
    _, rejects = check_written({"scenarios": [scenario]}, [line], sources.library)
    assert len(rejects) == 1 and "no one asks for" in rejects[0][1][-1]
