from pathlib import Path

import numpy as np

from eve.data.datagen.examples import REMOVAL_UNSUPPORTED_SHARE, datagen_examples, load_approved
from eve.data.datagen.shard import GenTool, Row, new_shard
from eve.data.generate.build import build
from eve.data.generate.library import load_specs

ROOT = Path(__file__).parents[1]
PEST = GenTool.from_dict(
    {
        "id": "t01",
        "topic": "garden > pests",
        "names": ["identify_garden_pest", "name_plant_bug", "spot_leaf_eater"],
        "descriptions": ["Say which insect is damaging a plant", "Tell what creature bites leaves"],
        "args": [
            {"name": "symptom", "type": "str"},
            {"name": "kind", "type": "enum", "options": ["indoor", "outdoor"], "required": False},
        ],
    }
)
SUM = GenTool.from_dict(
    {
        "id": "t02",
        "topic": "math > sums",
        "names": ["sum_pair", "total_two"],
        "descriptions": ["Put two values together", "Combine a pair of amounts"],
        "args": [{"name": "a", "type": "float"}, {"name": "b", "type": "float"}],
        "capability": "add",  # a twin of the training tool add
    }
)


def test_a_request_is_only_shown_with_a_name_and_description_it_avoids():
    row = Row("t01-r01", "t01", "what insect made these holes", "call", {"symptom": "holes"})
    specs = load_specs(ROOT / "data/tools")
    out = datagen_examples([PEST], [row] * 200, specs, np.random.default_rng(0))
    shown = {
        (t["name"], t["description"])
        for ex in out
        for t in ex["tools"]
        if t["name"] in PEST.names and ex["intent"] == "call"
    }
    assert {d for _, d in shown} == {PEST.descriptions[1]}  # "insect" is in the first one
    assert {n for n, _ in shown} == set(PEST.names)


def pest_row(i, intent="call"):
    args = {"symptom": "?" if intent == "ask" else f"holes{i}", "kind": "outdoor"}
    utterance = (
        f"my kale has holes{i}" if intent != "ask" else f"something is wrong with my kale {i}"
    )
    return Row(f"t01-r{i:04d}", "t01", utterance, intent, None if intent == "unsupported" else args)


def test_datagen_examples_rename_calls_and_block_twins():
    specs = load_specs(ROOT / "data/tools")
    rows = [pest_row(i) for i in range(800)] + [pest_row(i, "ask") for i in range(800, 900)]
    rows += [pest_row(i, "unsupported") for i in range(900, 950)]
    rows += [
        Row(
            f"t02-r{i:03d}", "t02", f"{i} and {i + 1} together please", "call", {"a": i, "b": i + 1}
        )
        for i in range(100)
    ]
    out = datagen_examples([PEST, SUM], rows, specs, np.random.default_rng(0))
    assert len(out) == len(rows) and all(r["source"] == "datagen" for r in out)
    shown_names = set()
    removed = 0
    for row, ex in zip(rows, out, strict=True):
        names = [t["name"] for t in ex["tools"]]
        assert 1 <= len(names) <= 5 and len(set(names)) == len(names)
        if row.tool == "t02":
            assert "add" not in names  # the twin's capability never shares a prompt
        if ex["intent"] == "unsupported" and row.intent == "call":
            removed += 1
            gold = PEST if row.tool == "t01" else SUM
            assert not set(names) & set(gold.names)
            continue
        if ex["calls"]:
            call = ex["calls"][0]
            assert call["tool"] in names and call["args"] == row.args
            shown_names.add(call["tool"])
        assert ex["intent"] == row.intent
    assert shown_names >= {"identify_garden_pest", "name_plant_bug", "spot_leaf_eater"}
    calls = sum(r.intent == "call" for r in rows)
    assert abs(removed / calls - REMOVAL_UNSUPPORTED_SHARE) < 0.03


def _lines(tmp_path, name):
    gen = tmp_path / "generation"
    for line_file in (
        "chitchat.txt",
        "unsupported.txt",
        "heldout/chitchat.txt",
        "heldout/unsupported.txt",
    ):
        (gen / line_file).parent.mkdir(parents=True, exist_ok=True)
        (gen / line_file).write_text((ROOT / "data/generation" / line_file).read_text())
    return gen


def _build(tmp_path, name, datagen_dir=None):
    out = tmp_path / name
    build(
        out / "task",
        out / "evals",
        300,
        50,
        seed=0,
        tools_dir=ROOT / "data/tools",
        generation_dir=_lines(tmp_path, name),
        external_dir=tmp_path / "no-external",
        novel_dir=ROOT / "data/novel-tools",
        datagen_dir=datagen_dir,
    )
    files = sorted((out / "task").glob("*.jsonl")) + sorted((out / "evals").glob("*.jsonl"))
    return {p.relative_to(out).as_posix(): p.read_text() for p in files}


def _shard(root, status, rows):
    topics = [f"area{i} > sub{i}" for i in range(10)]
    shard = new_shard(root, topics)
    shard.save_tools([PEST])
    shard.save_rows(rows)
    shard.set_status(status)


def test_build_without_datagen_is_unchanged(tmp_path):
    (tmp_path / "empty").mkdir()
    plain = _build(tmp_path, "a")
    assert _build(tmp_path, "b") == plain
    assert _build(tmp_path, "c", tmp_path / "empty") == plain


def test_unsupported_rows_never_offer_a_tool_close_to_the_gold_one():
    """Leaving the gold tool out must not leave a near neighbour that could serve the request:
    same area, or sharing a word with the gold tool."""
    cousin = GenTool.from_dict(
        {
            "id": "t03",
            "topic": "garden > weeds",
            "names": ["pull_weeds", "clear_weed_patch"],
            "descriptions": ["Say how to clear weeds from a plant bed", "Plan weeding"],
        }
    )
    specs = load_specs(ROOT / "data/tools")
    rows = [pest_row(i) for i in range(400)] + [pest_row(i, "unsupported") for i in range(400, 600)]
    out = datagen_examples([PEST, cousin], rows, specs, np.random.default_rng(1))
    bank_words = {
        "insect",
        "damaging",
        "plant",
        "creature",
        "bites",
        "leaves",
        "garden",
        "pest",
        "bug",
    }
    seen = 0
    for ex in out:
        if ex["intent"] != "unsupported":
            continue
        others = [t for t in ex["tools"] if t["name"] not in PEST.names]
        seen += len(others)
        assert all(t["name"] != "pull_weeds" for t in others)
        for t in others:
            words = set((t["name"].replace("_", " ") + " " + t["description"]).lower().split())
            assert not words & bank_words, t
    assert seen > 100


def test_build_refuses_a_missing_datagen_directory(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="no-such-dir"):
        _build(tmp_path, "a", tmp_path / "no-such-dir")


def test_build_with_datagen_keeps_evals_identical(tmp_path):
    shards = tmp_path / "shards"
    _shard(shards, "approved", [pest_row(i) for i in range(40)])
    _shard(shards, "flagged", [pest_row(i) for i in range(40, 80)])
    tools, rows = load_approved(shards)
    assert [t.names for t in tools] == [PEST.names] and tools[0].id == "0001-t01"
    assert [r.id for r in rows] == [pest_row(i).id for i in range(40)]
    assert {r.tool for r in rows} == {"0001-t01"}
    plain = _build(tmp_path, "a")
    with_datagen = _build(tmp_path, "b", shards)
    evals = {k: v for k, v in plain.items() if k.startswith("evals/")}
    assert {k: v for k, v in with_datagen.items() if k.startswith("evals/")} == evals
    train = with_datagen["task/train.jsonl"]
    assert train.startswith(plain["task/train.jsonl"])
    assert train.count('"source": "datagen"') == 40


def test_build_keeps_one_copy_of_a_request_two_shards_wrote(tmp_path):
    shards = tmp_path / "shards"
    _shard(shards, "approved", [pest_row(0)])
    _shard(shards, "approved", [pest_row(0)])  # written at the same time by another process
    train = _build(tmp_path, "a", shards)["task/train.jsonl"]
    assert train.count('"source": "datagen"') == 1
