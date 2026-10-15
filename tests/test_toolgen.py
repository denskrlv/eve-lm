from collections import Counter
from pathlib import Path

from colibri.data.generate.library import load_specs
from colibri.data.generate.toolgen import generated_specs

ROOT = Path(__file__).parents[1]


def test_generated_tools():
    specs = generated_specs()
    assert len(specs) >= 200
    assert generated_specs() == specs  # deterministic
    names = [s.name for s in specs]
    hand = {
        s.name for s in load_specs(ROOT / "data/tools") + load_specs(ROOT / "data/tools/heldout")
    }
    assert len(set(names)) == len(names) and not set(names) & hand
    by_capability = Counter(s.capability for s in specs)
    assert max(by_capability.values()) <= 2 and len(by_capability) >= 100


def test_generated_tools_avoid_held_out_topics():
    words = ("flight", "hotel", "currency", "add_numbers", "multiply", "divide", "convert")
    for spec in generated_specs():
        text = spec.tool.render().lower()
        assert not any(w in text for w in words), spec.name
