from pathlib import Path

import numpy as np

from eve.data.generate.build import Sources, build, template_examples
from eve.data.generate.sampler import ToolBank
from eve.data.generate.validate import normalize
from eve.data.jsonl import read_jsonl

ROOT = Path(__file__).parents[1]
LINE_FILES = [f"{d}{n}.txt" for d in ("", "heldout/") for n in ("chitchat", "unsupported")]


def test_build(tmp_path):
    gen = tmp_path / "generation"
    for name in LINE_FILES:
        (gen / name).parent.mkdir(parents=True, exist_ok=True)
        (gen / name).write_text((ROOT / "data/generation" / name).read_text())

    report = build(
        tmp_path / "task",
        tmp_path / "evals",
        300,
        50,
        seed=0,
        tools_dir=ROOT / "data/tools",
        generation_dir=gen,
        external_dir=tmp_path / "no-external",
        novel_dir=ROOT / "data/novel-tools",
    )
    removed = report["removed_duplicates"]
    assert report["splits"]["train"]["examples"] == 300 - sum(
        n for where, n in removed.items() if where.startswith("train ")
    )
    assert report["splits"]["val"]["examples"] == 0  # MASSIVE dev only, and none here
    assert sorted(p.name for p in (tmp_path / "task").iterdir()) == ["report.json", "train.jsonl"]
    train = {normalize(r["utterance"]) for r in read_jsonl(tmp_path / "task/train.jsonl")}
    for path in (tmp_path / "evals").glob("*.jsonl"):  # every scored set stays unseen
        assert not train & {normalize(r["utterance"]) for r in read_jsonl(path)}, path.name
    heldout = read_jsonl(tmp_path / "evals/heldout-templates.jsonl")
    assert len(heldout) == 50
    novel = read_jsonl(tmp_path / "evals/novel-tools.jsonl")
    assert {r.get("group") for r in novel} == {"shared word", "no shared word", None}
    assert all(r["source"] == "novel" for r in novel)


def test_template_small_talk_is_never_repeated():
    """NONE rows get new random tools each time, so only the words can tell a repeat."""
    sources = Sources.load(ROOT / "data/tools", ROOT / "data/generation")
    rng = np.random.default_rng(0)
    rows = template_examples(ToolBank(sources.train), 3000, rng, sources)
    none = [normalize(r["utterance"]) for r in rows if r["intent"] == "none"]
    assert none and len(none) == len(set(none))
