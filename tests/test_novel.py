import json
from pathlib import Path

import numpy as np
import pytest

from eve.data.generate.build import Sources
from eve.data.generate.novel import (
    SHARED,
    UNSHARED,
    load_novel,
    novel_examples,
    shares_word,
    stem,
)
from eve.data.schema import SchemaError

ROOT = Path(__file__).parents[1]
NOVEL = ROOT / "data/novel-tools"


@pytest.fixture(scope="module")
def library():
    return Sources.load(ROOT / "data/tools", ROOT / "data/generation").library


def test_a_shared_word_ignores_endings_but_not_spelling():
    assert stem("boxes") == "box" and stem("spelling") == "spell" and stem("bus") == "bus"
    assert shares_word("I want a stretching routine", ["stretch_routine"])
    assert shares_word("Check the tires", ["check_tire_pressure"])
    assert not shares_word("are my tyres ok", ["check_tire_pressure"])
    assert not shares_word("set the alarm to night mode", ["arm_security_system"])


def test_the_benchmark_is_valid_and_new(library):
    tools, requests = load_novel(NOVEL)
    assert 30 <= len(tools) <= 50 and not set(tools) & set(library)
    rows = novel_examples(np.random.default_rng(0), library, NOVEL)
    assert len(rows) == len(requests)
    assert rows == novel_examples(np.random.default_rng(0), library, NOVEL)  # seeded
    groups = [r["group"] for r in rows if r["calls"]]
    assert {SHARED, UNSHARED} == set(groups)
    assert groups.count(UNSHARED) >= len(groups) * 0.4
    for row in rows:
        gold = {c["tool"] for c in row["calls"]}
        shown = {t["name"]: t for t in row["tools"]}
        assert gold <= set(shown) and 1 <= len(shown) <= 5
        gold_domains = {tools[name].domain for name in gold}
        assert not {tools[name].domain for name in set(shown) - gold} & gold_domains


def test_a_tool_the_library_has_is_refused(library, tmp_path):
    (tmp_path / "tools").mkdir()
    tool = {"name": "set_lights", "description": "Turn lights on or off", "args": []}
    (tmp_path / "tools/home.json").write_text(json.dumps({"domain": "x", "tools": [tool]}))
    request = {"utterance": "lights please", "intent": "call", "calls": [{"tool": "set_lights"}]}
    (tmp_path / "requests.jsonl").write_text(json.dumps(request) + "\n")
    with pytest.raises(SchemaError, match="set_lights is in the training library"):
        novel_examples(np.random.default_rng(0), library, tmp_path)
