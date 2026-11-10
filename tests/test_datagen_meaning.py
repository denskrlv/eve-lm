from pathlib import Path

import numpy as np

from eve.data.datagen.meaning import MeaningIndex, blocked_index, library_index
from eve.data.generate.library import load_specs

ROOT = Path(__file__).parents[1]
AXES = {"sun": [1.0, 0.0, 0.0], "rays": [0.9, 0.1, 0.0], "fan": [0.0, 1.0, 0.0]}


def fake_embed(texts):
    """Known words map to fixed directions; anything else points along the third axis."""
    rows = [AXES.get(t, [0.0, 0.0, 1.0]) for t in texts]
    vectors = np.asarray(rows, dtype=np.float32).reshape(len(rows), 3)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_nearest_is_max_over_all_text_pairs():
    index = MeaningIndex(fake_embed)
    index.add("uv", ["sun", "other"])
    index.add("air", ["fan"])
    key, score = index.nearest(["rays", "fan"])
    assert key in ("uv", "air")
    assert index.nearest(["rays"])[0] == "uv"
    assert np.isclose(index.nearest(["rays"])[1], 0.9 / np.hypot(0.9, 0.1))
    assert np.isclose(index.nearest(["fan", "zzz"])[1], 1.0)


def test_empty_index_returns_none():
    assert MeaningIndex(fake_embed).nearest(["sun"]) is None


def test_blocked_index_has_the_41_novel_and_4_heldout_tools():
    index = blocked_index(fake_embed, ROOT / "data/novel-tools", ROOT / "data/tools/heldout")
    assert len(index) == 45
    assert "check_uv_index" in index.keys and "search_flights" in index.keys


def test_library_index_keys_by_tool_name():
    specs = load_specs(ROOT / "data/tools")[:5]
    index = library_index(fake_embed, specs)
    assert index.keys == [s.name for s in specs]


def test_library_index_embeds_in_one_call():
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    library_index(counting, load_specs(ROOT / "data/tools"))
    assert len(calls) == 1
