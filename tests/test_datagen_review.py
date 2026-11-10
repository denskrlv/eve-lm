import json

import pytest

from eve.data.datagen.llm import LLMError, Reply
from eve.data.datagen.requests import RowStats
from eve.data.datagen.review import (
    APPROVE_FIX_RATE,
    REVIEW_SCHEMA,
    apply_review,
    review_message,
    review_shard,
    shard_report,
)
from eve.data.datagen.shard import GenTool, Row
from eve.data.datagen.tools import ToolStats
from eve.data.schema import Tool

PEST = GenTool.from_dict(
    {
        "id": "t01",
        "topic": "garden > pests",
        "names": ["identify_garden_pest", "name_plant_bug"],
        "descriptions": ["Say which insect is damaging a plant", "Tell what creature bites leaves"],
        "args": [{"name": "symptom", "type": "str"}],
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
ROWS = [
    Row(
        "t01-r01", "t01", "something chews holes in my kale", "call", {"symptom": "holes"}, {}, "m"
    ),
    Row("t01-r02", "t01", "spots all over my roses", "call", {"symptom": "spots"}, {}, "m"),
    Row("t01-r03", "t01", "my tomato leaves curl", "call", {"symptom": "curl"}, {}, "m"),
    Row("t02-r01", "t02", "crank it up, I'm melting", "call", {"speed": "high"}, {}, "m"),
]


def test_fix_is_rechecked_and_marked_fixed():
    data = {"fixes": [{"id": "t01-r02", "utterance": "white spots all over my roses", "why": "x"}]}
    result = apply_review(data, [PEST, FAN], ROWS)
    fixed = next(r for r in result.rows if r.id == "t01-r02")
    assert fixed.utterance == "white spots all over my roses" and fixed.writer == "fixed"
    assert fixed.args == {"symptom": "spots"}
    assert result.fixed == 1 and result.dropped == 0 and result.reviewed == 4
    assert {r.writer for r in result.rows if r.id != "t01-r02"} == {"m"}


def test_a_fix_that_changes_nothing_is_not_counted():
    same = {"id": "t01-r02", "utterance": " spots all over my roses", "why": "keep"}
    result = apply_review({"fixes": [same, {"id": "t01-r03", "why": "keep"}]}, [PEST, FAN], ROWS)
    assert result.fixed == 0 and result.dropped == 0 and result.fix_rate == 0
    assert {r.writer for r in result.rows} == {"m"}
    assert [r["action"] for r in result.reasons] == ["unchanged", "unchanged"]


def test_fix_failing_checks_is_dropped():
    data = {
        "fixes": [
            {"id": "t01-r02", "utterance": "which insect bites spots into the leaves", "why": "x"}
        ]
    }
    result = apply_review(data, [PEST, FAN], ROWS)
    assert "t01-r02" not in {r.id for r in result.rows}
    assert result.fixed == 0 and result.dropped == 1
    assert any("insect" in r["why"] for r in result.reasons)


def test_review_never_edits_a_tool_and_ignores_unknown_ids():
    data = {
        "fixes": [
            {"id": "t02-r01", "args": {"colour": "red"}, "why": "x"},
            {"id": "t09-r01", "utterance": "who knows", "why": "x"},
            {"id": "t01-r01", "utterance": "worms in my kale", "args": {"symptom": "worms"}},
        ],
        "drop_rows": [{"id": "t01-r03", "why": "odd"}],
        "drop_tools": [{"id": "t01", "why": "broken"}],
    }
    result = apply_review(data, [PEST, FAN], ROWS)
    assert [t.id for t in result.tools] == ["t02"]
    assert result.tools[0] == FAN  # untouched
    assert result.rows == []  # t02-r01's fix named an arg FAN lacks: dropped
    assert result.tools_dropped == ["t01"]
    assert result.reviewed == 4 and result.fixed == 0 and result.dropped == 4  # 3 with t01
    assert any("unknown" in r["why"] for r in result.reasons)


def test_fix_rate_and_approval():
    tools = [PEST]
    rows = [
        Row(f"t01-r{i:03d}", "t01", f"holes number {i} in my kale", "call", {"symptom": "holes"})
        for i in range(500)
    ]
    drops = [{"id": r.id, "why": "x"} for r in rows[:50]]
    result = apply_review({"drop_rows": drops}, tools, rows)
    assert result.fix_rate == pytest.approx(0.10)
    assert not shard_report(ToolStats(), RowStats(), result)["approved"]
    result = apply_review({"drop_rows": drops[:49]}, tools, rows)
    report = shard_report(
        ToolStats(tokens=100, seconds=4.0), RowStats(tokens=60, seconds=3.0), result
    )
    assert report["approved"] and result.fix_rate < APPROVE_FIX_RATE
    assert report["speed"] == {"tools_tokens_per_s": 25.0, "requests_tokens_per_s": 20.0}
    json.dumps(report)


def test_review_shard_sends_tools_rows_and_blocked_and_rejects_bad_replies():
    seen = {}

    def claude(system, prompt, schema):
        seen["prompt"], seen["schema"] = prompt, schema
        return Reply("", {"fixes": []}, cost=0.4)

    blocked = [Tool("check_uv_index", "Say how strong the sun is today")]
    result = review_shard(claude, [PEST, FAN], ROWS, blocked)
    assert result.cost == 0.4 and len(result.rows) == 4
    assert "check_uv_index" in seen["prompt"] and "t02-r01" in seen["prompt"]
    assert seen["schema"] == REVIEW_SCHEMA
    assert "identify_garden_pest" in review_message([PEST], ROWS[:1], blocked)
    with pytest.raises(LLMError):
        review_shard(lambda s, p, sc: Reply("nope", None), [PEST], ROWS[:1], blocked)


def test_the_review_knows_the_rules_the_code_checks():
    from eve.data.datagen.review import REVIEW_SYSTEM

    text = " ".join(REVIEW_SYSTEM.split())
    assert "may be written as a word" in text  # 7 or seven: don't "fix" number words
    assert "at least one of the tool's names and one of its descriptions" in text
    assert "only the words for that argument" in text  # value edges
    assert "never on an optional argument" in text


def test_junk_in_a_fix_is_dropped_never_a_crash():
    junk = ["thirty", "", [30], {"n": 30}, True, float("nan"), None, "?", 5]
    for value in junk:
        for intent in ("call", "ask", "unsupported", "none", 7, None):
            fix = {"id": "t02-r01", "intent": intent, "args": {"speed": value}, "utterance": value}
            result = apply_review({"fixes": [fix]}, [PEST, FAN], ROWS)
            assert result.fixed + result.dropped == 1


def test_dropping_a_tool_counts_its_requests_against_approval():
    rows = [
        Row(f"t01-r{i:02d}", "t01", f"holes number {i} in my kale", "call", {"symptom": "holes"})
        for i in range(9)
    ] + [ROWS[3]]
    result = apply_review({"drop_tools": [{"id": "t01", "why": "reserved"}]}, [PEST, FAN], rows)
    assert result.rows == [ROWS[3]]
    assert result.fix_rate == pytest.approx(0.9)
    assert not shard_report(ToolStats(), RowStats(), result)["approved"]
