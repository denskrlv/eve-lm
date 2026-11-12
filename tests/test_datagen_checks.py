from eve.data.datagen.checks import clean_pairs, row_problems, shared_words, tool_problems
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
    shared = row_problems(row("which insect bites the leaves, holes everywhere"), PEST, set())
    assert any('"insect"' in p and "every" in p for p in shared)
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


def test_a_request_needs_only_one_clean_name_and_description():
    one_word = row("what insect made these holes")  # "insect" is only in the first description
    assert row_problems(one_word, PEST, set()) == []
    assert clean_pairs(one_word, PEST) == [(0, 1), (1, 1)]
    assert clean_pairs(row("something chews holes in my kale"), PEST) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    plant = row("my plant has holes")  # in name_plant_bug and the first description
    assert clean_pairs(plant, PEST) == [(0, 1)]


def test_ask_may_only_leave_out_a_required_value():
    optional = Row("t01-r05", "t01", "my kale has holes", "ask", {"symptom": "holes", "crop": "?"})
    assert any("required" in p for p in row_problems(optional, PEST, set()))
    required = Row("t01-r06", "t01", "something is wrong with my kale", "ask", {"symptom": "?"})
    assert row_problems(required, PEST, set()) == []


def test_long_descriptions_are_a_tool_problem():
    long = "Say which small insect or other creature is slowly damaging a plant in the garden"
    wordy = GenTool("t", "a > b", PEST.names, (long, PEST.descriptions[1]))
    assert any("words" in p for p in tool_problems(wordy, set()))


def test_text_values_keep_the_case_of_the_request():
    lower = Row("t01-r07", "t01", "something chews Holes in my kale", "call", {"symptom": "holes"})
    assert any("case" in p for p in row_problems(lower, PEST, set()))


def test_ask_leaves_out_only_text_or_number_values():
    timer = GenTool.from_dict(
        {
            "id": "t03",
            "topic": "home > air",
            "names": ["fan_timer", "breeze_clock"],
            "descriptions": ["Stops the fan after a while", "Sets when cooling ends"],
            "args": [{"name": "minutes", "type": "int"}, {"name": "on", "type": "bool"}],
        }
    )
    left_out = Row("t03-r01", "t03", "Stop the breeze in 20 min", "ask", {"minutes": 20, "on": "?"})
    assert any("text or number" in p for p in row_problems(left_out, timer, set()))


def test_an_optional_enum_needs_the_request_to_say_it():
    hair = GenTool.from_dict(
        {
            "id": "t04",
            "topic": "hair > styles",
            "names": ["find_hair_tutorial", "show_style_steps"],
            "descriptions": ["Show videos on styling hair", "Teach the steps for a look"],
            "args": [
                {"name": "style_name", "type": "str"},
                {
                    "name": "difficulty",
                    "type": "enum",
                    "options": ["easy", "hard"],
                    "required": False,
                },
            ],
        }
    )
    invented = Row(
        "t04-r01",
        "t04",
        "walk me through a messy bun",
        "call",
        {"style_name": "messy bun", "difficulty": "easy"},
    )
    assert any("difficulty" in p for p in row_problems(invented, hair, set()))
    said = Row(
        "t04-r02",
        "t04",
        "an easy messy bun, walk me through it",
        "call",
        {"style_name": "messy bun", "difficulty": "easy"},
    )
    assert row_problems(said, hair, set()) == []
    required = Row("t02-r01", "t02", "make the room feel colder", "call", {"speed": "high"})
    assert row_problems(required, FAN, set()) == []  # a required one may be implied


def test_a_text_value_names_the_thing_not_a_pointer_or_the_argument():
    for value, utterance in [
        ("this crop", "something chews holes in this crop"),
        ("listed here", "fix the holes listed here"),
        ("the symptom", "check the symptom on my kale"),
    ]:
        filler = Row("t01-r08", "t01", utterance, "call", {"symptom": value})
        assert any(repr(value) in p for p in row_problems(filler, PEST, set())), value
