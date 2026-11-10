import json

import numpy as np

from eve.data.datagen.llm import Reply
from eve.data.datagen.requests import CARDS, draw_cards, write_requests
from eve.data.datagen.shard import GenTool

CONSTRAINTS = {
    "speaker": ["a nurse", "a teen"],
    "situation": ["driving"],
    "form": ["question", "command"],
    "length": ["3-6 words"],
    "tone": ["casual"],
}
PEST = GenTool.from_dict(
    {
        "id": "t03",
        "topic": "garden > pests",
        "names": ["identify_garden_pest", "name_plant_bug"],
        "descriptions": ["Say which insect is damaging a plant", "Tell what creature bites leaves"],
        "args": [{"name": "symptom", "type": "str"}],
    }
)
FAN = GenTool.from_dict(
    {
        "id": "t04",
        "topic": "home > air",
        "names": ["fan_speed", "set_breeze"],
        "descriptions": ["Choose how fast the fan spins", "Pick the strength of moving air"],
        "args": [{"name": "speed", "type": "enum", "options": ["low", "high"]}],
    }
)


class Script:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, system, prompt, schema):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else {"rows": []}
        return Reply("", reply, 7, 0.5)


def cards(*intents):
    return [{"intent": i, "speaker": "a nurse", "form": "question"} for i in intents]


def test_cards_follow_the_counts_and_skip_ask_without_required_text_or_number():
    rng = np.random.default_rng(0)
    drawn = draw_cards(PEST, CONSTRAINTS, rng)
    assert [c["intent"] for c in drawn].count("call") == CARDS["call"]
    assert [c["intent"] for c in drawn].count("ask") == CARDS["ask"]
    assert [c["intent"] for c in drawn].count("near") == CARDS["near"]
    assert set(drawn[0]) == {"intent", *CONSTRAINTS}
    assert {c.get("leave_out") for c in drawn if c["intent"] == "ask"} == {"symptom"}
    assert all("leave_out" not in c for c in drawn if c["intent"] != "ask")
    fan = [c["intent"] for c in draw_cards(FAN, CONSTRAINTS, rng)]
    assert "ask" not in fan and fan.count("call") == CARDS["call"] + CARDS["ask"]


def test_write_requests_resends_only_rejected_cards_with_reasons():
    first = {
        "rows": [
            {
                "card": 1,
                "utterance": "something chews holes in my kale",
                "args": {"symptom": "holes"},
            },
            {
                "card": 2,
                "utterance": "which insect bites holes in the leaves",
                "args": {"symptom": "holes"},
            },
        ]
    }
    second = {
        "rows": [
            {"card": 2, "utterance": "who made holes in my roses", "args": {"symptom": "holes"}}
        ]
    }
    script = Script(first, second)
    rows, stats = write_requests(script, PEST, cards("call", "call"), set(), "m")
    assert [r.id for r in rows] == ["t03-r01", "t03-r02"]
    assert all(r.writer == "m" and r.intent == "call" for r in rows)
    assert stats.rejected["shared"] == 1 and stats.rounds == {1: 1, 2: 1}
    lines = [json.loads(line) for line in script.prompts[1].splitlines() if line.startswith("{")]
    assert [line["card"] for line in lines] == [2]
    assert (
        lines[0]["rejected"] == "which insect bites holes in the leaves"
        and "insect" in lines[0]["because"][0]
    )
    assert stats.written == 3 and stats.tokens == 14  # attempts, kept or not


def test_missing_extra_and_unknown_cards_are_rejects():
    first = {
        "rows": [
            {"card": 1, "utterance": "holes all over my kale", "args": {"symptom": "holes"}},
            {"card": 1, "utterance": "again holes in the kale", "args": {"symptom": "holes"}},
            {"card": 9, "utterance": "holes in lettuce", "args": {"symptom": "holes"}},
            "not a row",
        ]
    }
    script = Script(
        first,
        {"rows": [{"card": 2, "utterance": "spots on my roses", "args": {"symptom": "spots"}}]},
    )
    rows, stats = write_requests(script, PEST, cards("call", "call"), set(), "m")
    assert len(rows) == 2
    assert stats.rejected["missing"] == 1 and stats.rejected["extra"] == 3
    assert "not written" in script.prompts[1]


def test_near_rows_are_unsupported_without_calls():
    reply = {
        "rows": [{"card": 1, "utterance": "which insect spray is safe for a plant", "args": {}}]
    }
    rows, _ = write_requests(Script(reply), PEST, cards("near"), set(), "m")
    assert rows[0].intent == "unsupported" and rows[0].args is None


def test_gives_up_after_rounds_and_repeats_count():
    reply = {"rows": [{"card": 1, "utterance": "holes in my kale", "args": {"symptom": "holes"}}]}
    seen = {"holes in my kale"}
    script = Script(*[reply] * 5)
    rows, stats = write_requests(script, PEST, cards("call"), seen, "m")
    assert rows == [] and stats.dropped == 1 and stats.rejected["repeat"] == 3
    assert len(script.prompts) == 3


def test_each_card_lists_the_words_of_one_name_and_description_to_avoid():
    script = Script({"rows": []})
    write_requests(script, PEST, cards("call", "call", "near"), set(), "m")
    lines = [json.loads(line) for line in script.prompts[0].splitlines() if line.startswith("{")]
    first, second, near = (line.get("avoid") for line in lines)
    assert "identify" in first and "insect" in first and "the" not in first
    assert "bug" in second and "bites" in second and "insect" not in second
    assert near is None  # a near miss may use the tool's words


def test_every_reject_is_logged_with_its_reasons():
    first = {
        "rows": [
            {
                "card": 1,
                "utterance": "which insect bites holes in the leaves",
                "args": {"symptom": "holes"},
            },
            {"card": 9, "utterance": "stray", "args": {}},
        ]
    }
    script = Script(
        first,
        {
            "rows": [
                {"card": 1, "utterance": "holes all over my kale", "args": {"symptom": "holes"}}
            ]
        },
    )
    _, stats = write_requests(script, PEST, cards("call"), set(), "m")
    shared, extra = stats.log
    assert shared["tool"] == "t03" and shared["card"] == 1 and shared["round"] == 1
    assert (
        shared["utterance"] == "which insect bites holes in the leaves"
        and shared["kind"] == "shared"
    )
    assert "every name" in shared["because"][0]
    assert extra["kind"] == "extra"
    assert "log" not in stats.to_dict()


ASK_CARD = [{"intent": "ask", "speaker": "a nurse", "form": "question", "leave_out": "symptom"}]


def ask_reply(utterance, args):
    return {"rows": [{"card": 1, "utterance": utterance, "args": args}]}


def test_an_ask_leaves_out_the_value_its_card_names_and_code_checks_it_is_absent():
    says_it = ask_reply("what made holes in my kale", {"symptom": "holes"})
    leaves_it = ask_reply("something is wrong with my kale", {"symptom": "spots"})
    rows, stats = write_requests(Script(says_it, leaves_it), PEST, ASK_CARD, set(), "m")
    assert rows[0].intent == "ask" and rows[0].args == {"symptom": "?"}
    assert stats.rejected["ask"] == 1 and "holes" in stats.log[0]["because"][0]


def test_an_ask_without_a_value_for_the_left_out_argument_is_a_reject():
    replies = [
        ask_reply("my kale is sad, do something", {}),
        ask_reply("my kale is sad, do something", {"symptom": "?"}),
    ]
    rows, stats = write_requests(Script(*replies), PEST, ASK_CARD, set(), "m")
    assert rows[0].args == {"symptom": "?"}  # a "?" given directly is still fine
    assert stats.rejected["schema"] == 1 and "symptom" in stats.log[0]["because"][0]


def test_the_prompt_tells_an_ask_card_which_value_to_leave_out():
    script = Script({"rows": []})
    write_requests(script, PEST, ASK_CARD, set(), "m")
    line = next(json.loads(x) for x in script.prompts[0].splitlines() if x.startswith("{"))
    assert line["leave_out"] == "symptom"


def test_null_args_mean_none_given():
    no_args = GenTool.from_dict({**FAN.to_dict(), "args": []})
    reply = {"rows": [{"card": 1, "utterance": "make it breezy in here", "args": None}]}
    rows, _ = write_requests(Script(reply), no_args, cards("call"), set(), "m")
    assert rows[0].args == {}
    optional = {"name": "crop", "type": "str", "required": False}
    with_optional = GenTool.from_dict(
        {**PEST.to_dict(), "args": [*PEST.to_dict()["args"], optional]}
    )
    args = {"symptom": "holes", "crop": None}
    reply = {"rows": [{"card": 1, "utterance": "something chews holes in my kale", "args": args}]}
    rows, stats = write_requests(Script(reply), with_optional, cards("call"), set(), "m")
    assert rows[0].args == {"symptom": "holes"} and not stats.rejected


def test_a_value_in_the_wrong_case_goes_back_as_a_value_reject():
    reply = {
        "rows": [
            {
                "card": 1,
                "utterance": "something chews Holes in my kale",
                "args": {"symptom": "holes"},
            }
        ]
    }
    _, stats = write_requests(Script(reply), PEST, cards("call"), set(), "m")
    assert stats.rejected["value"] == 1


def test_a_tool_named_with_case_does_not_turn_schema_rejects_into_value_rejects():
    from eve.data.datagen.requests import _kind

    assert _kind(["call to check_case_status: speed: None is not a valid str"]) == "schema"
    assert _kind(["symptom='holes' must be written with the request's case"]) == "value"
