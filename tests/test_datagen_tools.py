import numpy as np

from eve.data.datagen.llm import Reply
from eve.data.datagen.meaning import MeaningIndex
from eve.data.datagen.tools import ROUNDS, TOOLS_SYSTEM, Judge, invent_tools

# A text's meaning in the fake: its first word, plus its theme when it has a theme word. Two
# texts with the same first word and theme are the same; one shared part makes them related.
THEMES = {
    "garden": {"plant", "greenery", "hedge", "shrubs", "water", "soil"},
    "sun": {"sun", "skin"},
    "math": {"numbers", "values"},
}
AXIS: dict[str, int] = {}


def fake_embed(texts):
    rows = np.zeros((len(texts), 256), dtype=np.float32)
    for i, text in enumerate(texts):
        words = text.lower().split()
        rows[i, AXIS.setdefault("first:" + words[0], len(AXIS))] = 1.0
        for theme, keys in THEMES.items():
            if keys & set(words):
                rows[i, AXIS.setdefault("theme:" + theme, len(AXIS))] = 1.0
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def tool(names, descriptions, args=None):
    return {
        "names": names,
        "descriptions": descriptions,
        "args": args if args is not None else [{"name": "plant", "type": "str"}],
    }


GOOD = tool(["water_plant", "give_plant_drink"], ["Pour water on a plant", "Keep greenery moist"])
GOOD2 = tool(["trim_hedge", "shape_bushes"], ["Cut the hedge back", "Neaten overgrown shrubs"])
SUNBURN = tool(["sun_strength", "burn_risk"], ["Tell how strong the sun is", "Warn about skin"])
TWIN = tool(["sum_pair", "total_two"], ["Add two numbers", "Combine a pair of values"], [])


class Script:
    """A Generate that answers from a queue and records each prompt."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, system, prompt, schema):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else {"tools": []}
        if isinstance(reply, str):
            return Reply(reply, None, 5, 1.0)
        return Reply("", reply, 10, 1.0)


def judge():
    blocked = MeaningIndex(fake_embed)
    blocked.add("check_uv_index", ["Tell the uv index"])
    library = MeaningIndex(fake_embed)
    library.add("add", ["Add two numbers"])
    return Judge(blocked, MeaningIndex(fake_embed), library, {"add": "add"}, {"add"})


def run(script, per_topic=2, topics=("garden > plants",)):
    return invent_tools(script, list(topics), judge(), per_topic)


def test_the_prompt_asks_for_short_everyday_tools_without_seed_words():
    script = Script({"tools": [GOOD]})
    run(script, per_topic=1)
    assert "seed" not in script.prompts[0].lower()
    assert "10 words" in TOOLS_SYSTEM and "seed" not in TOOLS_SYSTEM.lower()


def test_invent_tools_accepts_good_tools_and_sends_rejects_back():
    bad_schema = tool(["Bad Name", "x_y"], ["A thing", "Another thing"])
    script = Script({"tools": [GOOD, SUNBURN]}, {"tools": [bad_schema]}, {"tools": [GOOD2]})
    tools, stats = run(script, per_topic=2)
    assert [t.names[0] for t in tools] == ["water_plant", "trim_hedge"]
    assert [t.id for t in tools] == ["t01", "t02"] and stats.kept == 2
    assert all(t.topic == "garden > plants" for t in tools)
    assert stats.proposed == 4 and stats.rejected["blocked"] == 1 and stats.rejected["schema"] == 1
    assert "check_uv_index" in script.prompts[1] and "water_plant" in script.prompts[1]
    assert "Asked for 1" in script.prompts[1] and "snake_case" in script.prompts[2]
    assert stats.tokens == 30 and stats.seconds == 3.0


def test_repeat_of_an_accepted_tool_is_rejected():
    again = tool(["moisten_plant", "plant_drink"], ["Pour some water", "Damp the soil"])
    tools, stats = run(Script({"tools": [GOOD, again]}, {"tools": [GOOD2]}), per_topic=2)
    assert stats.rejected["repeat"] == 1
    assert [t.names[0] for t in tools] == ["water_plant", "trim_hedge"]


def test_twin_is_kept_with_the_library_capability():
    tools, stats = run(Script({"tools": [TWIN]}), per_topic=1)
    assert tools[0].capability == "add" and stats.twins == 1


def test_unparseable_reply_counts_as_rejects_not_a_crash():
    tools, stats = run(Script("I can't do that", {"tools": "nope"}, {"tools": [GOOD]}), per_topic=1)
    assert [t.names[0] for t in tools] == ["water_plant"]
    assert stats.rejected["json"] == 2


def test_gives_up_after_rounds():
    script = Script(*[{"tools": [SUNBURN, SUNBURN]}] * 10)
    tools, stats = run(script, per_topic=2)
    assert tools == [] and stats.dropped == 2 and len(script.prompts) == ROUNDS


def test_taken_names_are_rejected():
    clash = tool(["add", "plus_them"], ["Put numbers together", "Sum numbers up"], [])
    tools, stats = run(Script({"tools": [clash]}, {"tools": [GOOD]}), per_topic=1)
    assert stats.rejected["name"] == 1 and tools[0].names[0] == "water_plant"


def test_descriptions_of_two_different_things_are_rejected():
    mixed = tool(
        ["plan_bake_cost", "check_room_light"], ["Plan what baking costs", "Check the room light"]
    )
    tools, stats = run(Script({"tools": [mixed]}, {"tools": [GOOD]}), per_topic=1)
    assert stats.rejected["incoherent"] == 1 and tools[0].names[0] == "water_plant"


def test_text_where_a_list_belongs_is_a_schema_reject():
    letters = {**GOOD, "names": "abc"}
    tools, stats = run(Script({"tools": [letters]}, {"tools": [GOOD]}), per_topic=1)
    assert stats.rejected["schema"] == 1 and tools[0].names[0] == "water_plant"


def test_extra_tools_beyond_the_ask_are_ignored():
    tools, stats = run(Script({"tools": [GOOD, GOOD2]}), per_topic=1)
    assert len(tools) == 1 and stats.proposed == 1
