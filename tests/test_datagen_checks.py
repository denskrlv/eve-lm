from eve.data.datagen.checks import row_problems, shared_words, tool_problems
from eve.data.datagen.shard import GenTool, Row

ADD = ["add", "Add two numbers"]
PEST = GenTool(
    "t01",
    "garden > pests",
    ("identify_garden_pest", "name_plant_bug"),
    ("Say which insect is damaging a plant", "Tell what creature bites the leaves"),
)
PEST = GenTool.from_dict(
    {
        **PEST.to_dict(),
        "args": [
            {"name": "symptom", "type": "str"},
            {"name": "crop", "type": "str", "required": False},
        ],
    }
)
FAN = GenTool.from_dict(
    {
        "id": "t02",
        "topic": "home > air",
        "names": ["fan_speed", "set_breeze"],
        "descriptions": ["Choose how fast the fan spins", "Pick the strength of moving air"],
        "args": [{"name": "speed", "type": "enum", "options": ["low", "high"]}],
    }
)


def row(utterance, intent="call", args=None, tool="t01"):
    if args is None and intent != "unsupported":
        args = {"symptom": "holes"}
    return Row(f"{tool}-r01", tool, utterance, intent, args)


def test_shared_words_ignores_stopwords_endings_and_values():
    assert shared_words("what's 4 plus 9", ADD, [4, 9]) == set()
    assert shared_words("add 4 and 9", ADD, [4, 9]) == {"add"}
    assert shared_words("adding up", ADD) == {"add"}
    assert shared_words("my kale has holes", ["kale pests"], ["kale"]) == set()
    assert shared_words("four and nine", ["four seasons"], [4]) == set()


def test_tool_problems():
    assert tool_problems(PEST, set()) == []
    one_name = GenTool("t", "a > b", ("only_name",), PEST.descriptions)
    assert any("names" in p for p in tool_problems(one_name, set()))
    twin_names = GenTool("t", "a > b", ("same", "same"), PEST.descriptions)
    assert any("names" in p for p in tool_problems(twin_names, set()))
    one_desc = GenTool("t", "a > b", PEST.names, ("Say which insect is damaging a plant",))
    assert any("descriptions" in p for p in tool_problems(one_desc, set()))
    close = GenTool(
        "t", "a > b", PEST.names, ("Say which bug eats a plant", "Say which bug eats the plant")
    )
    assert any("differ" in p for p in tool_problems(close, set()))
    assert any("name_plant_bug" in p for p in tool_problems(PEST, {"name_plant_bug"}))


def test_row_problems():
    assert row_problems(row("something chews holes in my kale"), PEST, set()) == []
    missing = row_problems(row("something chews my kale", args={"symptom": "holes"}), PEST, set())
    assert any("not in the utterance" in p for p in missing)
    shared = row_problems(row("what insect made these holes"), PEST, set())
    assert any('"insect"' in p for p in shared)
    ask = Row("t02-r01", "t02", "change how the room feels", "ask", {"speed": "?"})
    assert any("ask" in p for p in row_problems(ask, FAN, set()))
    seen = {"something chews holes in my kale"}
    assert any(
        "repeat" in p for p in row_problems(row("Something chews holes in my kale!"), PEST, seen)
    )
    near = row("which insect spray is safe for a plant", intent="unsupported")
    assert row_problems(near, PEST, set()) == []
    bad = row_problems(Row("t01-r09", "t01", "kale holes", "call", {"colour": "red"}), PEST, set())
    assert bad and "colour" in bad[0]
