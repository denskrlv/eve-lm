import json
from pathlib import Path

import numpy as np
import pytest

from eve.data.generate import external
from eve.data.generate.external import (
    HELD_OUT,
    MASSIVE_FILE,
    MASSIVE_TOOLS,
    XLAM_FILE,
    convert_external,
    convert_massive,
    convert_xlam,
    short,
    snake,
    unsupported_variant,
    xlam_tool,
)
from eve.data.jsonl import read_jsonl, write_jsonl
from eve.data.render import TaskExample

WEATHER = {
    "name": "getWeatherForecast",
    "description": "Get the forecast for a city. Uses the OpenWeather API.",
    "parameters": {
        "location": {"type": "str", "description": "City name"},
        "days": {"type": "int, optional", "description": "Days ahead", "default": 3},
        "units": {
            "type": "str, optional",
            "description": "metric or imperial",
            "default": "metric",
        },
    },
}
LISTY = {
    "name": "sum_list",
    "description": "Adds items",
    "parameters": {"xs": {"type": "List[int]"}},
}


def xlam_row(query, answers, tools=(WEATHER,)):
    return {"query": query, "answers": json.dumps(answers), "tools": json.dumps(list(tools))}


def test_helpers():
    assert snake("getWeatherForecast") == "get_weather_forecast"
    assert snake("v1-sticker.id") == "v1_sticker_id" and snake("3d_print") == "n_3d_print"
    assert short("Get the forecast. Uses an API.", 100) == "Get the forecast"
    assert short("word " * 40, 20) == "word word word word"
    assert HELD_OUT.search("Search for hotels") and not HELD_OUT.search("multiple items")


def test_xlam_tool():
    tool, names, defaults = xlam_tool(WEATHER)
    assert tool.render() == (
        "<tool> get_weather_forecast : Get the forecast for a city <arg> location : str : City name"
        " <arg> days ? : int : Days ahead <arg> units ? : str : metric or imperial"
    )
    assert names["location"] == "location" and defaults["days"] == 3
    assert xlam_tool(LISTY) is None


def test_convert_xlam():
    rng = np.random.default_rng(0)
    answers = [
        {
            "name": "getWeatherForecast",
            "arguments": {"location": "Oslo", "days": "5", "units": "metric"},
        }
    ]
    ex, reason = convert_xlam(xlam_row("Weather in Oslo for the next 5 days?", answers), rng)
    assert reason == "ok" and ex.intent == "call"
    # "5" became an int; units equals its default and wasn't said, so it is left out.
    assert ex.calls[0].args == {"location": "Oslo", "days": 5}

    answers[0]["arguments"]["days"] = 7  # an optional value the user never said
    ex, _ = convert_xlam(xlam_row("Weather in Oslo for the next 5 days?", answers), rng)
    assert ex.calls[0].args == {"location": "Oslo"}

    answers[0]["arguments"]["location"] = "Bergen"  # a required value the user never said
    assert convert_xlam(xlam_row("Weather in Oslo?", answers), rng) == (
        None,
        "value not in the request",
    )
    sums = [{"name": "sum_list", "arguments": {"xs": [1, 2]}}]
    assert convert_xlam(xlam_row("add 1 and 2", sums, [LISTY]), rng)[1] == (
        "list, dict or other complex argument"
    )
    hotel = xlam_row("add 1 and 2, then find hotels", sums, [LISTY])
    assert convert_xlam(hotel, rng)[1] == "held-out topic"


def test_unsupported_variant():
    rng = np.random.default_rng(0)
    answers = [{"name": "getWeatherForecast", "arguments": {"location": "Oslo"}}]
    ex, _ = convert_xlam(xlam_row("Weather in Oslo", answers), rng)
    other = TaskExample.from_json(
        {
            "utterance": "x",
            "intent": "none",
            "tools": [{"name": "play_song", "description": "Play a song", "args": []}],
        }
    )
    variant = unsupported_variant(ex, list(ex.tools) + list(other.tools), rng)
    assert variant.intent == "unsupported" and [t.name for t in variant.tools] == ["play_song"]


def test_convert_massive():
    rows = [
        {
            "partition": "train",
            "intent": "iot_hue_lightoff",
            "utt": "turn off the kitchen lights",
            "annot_utt": "turn off the [house_place : kitchen] lights",
        },
        {"partition": "dev", "intent": "general_greet", "utt": "hi there", "annot_utt": "hi there"},
        {
            "partition": "test",
            "intent": "transport_taxi",
            "utt": "get me a taxi",
            "annot_utt": "get me a taxi",
        },
        {
            "partition": "train",
            "intent": "alarm_set",
            "utt": "wake me at 7 or 8",
            "annot_utt": "wake me at [time : 7] or [time : 8]",
        },
    ] + [
        {
            "partition": "train",
            "intent": "iot_hue_lightoff",
            "utt": f"lights off in the {room}",
            "annot_utt": f"lights off in the [house_place : {room}]",
        }
        for room in ["hall", "bedroom", "office", "garage", "attic"]
    ]
    parts, dropped = convert_massive(rows, np.random.default_rng(0))
    first = parts["train"][0]
    assert first.intent == "call" and first.calls[0].tool == MASSIVE_TOOLS["iot_hue_lightoff"][0]
    assert first.calls[0].args == {"house_place": "kitchen"}
    assert parts["dev"][0].intent == "none" and parts["test"] == []
    assert (
        dropped["the same slot twice"] == 1
        and dropped["intent left out (held-out topic or too rare)"] == 1
    )
    for ex in parts["train"]:
        if ex.intent == "unsupported":  # never offers another lights tool
            assert not any(t.domain == "massive-iot" for t in ex.tools)


def test_inline_tools_round_trip():
    ex, _ = convert_xlam(
        xlam_row(
            "Weather in Oslo", [{"name": "getWeatherForecast", "arguments": {"location": "Oslo"}}]
        ),
        np.random.default_rng(0),
    )
    assert TaskExample.from_json(ex.to_json(inline=True)) == ex


def test_held_out_topics_in_the_request_are_dropped():
    rng = np.random.default_rng(0)
    answers = [{"name": "getWeatherForecast", "arguments": {"location": "Oslo"}}]
    assert convert_xlam(
        xlam_row("Weather in Oslo, in celsius, then book a hotel there", answers), rng
    ) == (
        None,
        "held-out topic",
    )
    rows = [
        {
            "partition": "train",
            "intent": "qa_factoid",
            "utt": "can you convert my this money into",
            "annot_utt": "can you convert my this money into",
        },
        {
            "partition": "train",
            "intent": "general_quirky",
            "utt": "which airline flies there",
            "annot_utt": "which airline flies there",
        },
    ]
    parts, dropped = convert_massive(rows, rng)
    assert parts["train"] == [] and dropped["held-out topic in the request"] == 2


LIGHTS_OFF = [
    {
        "partition": "train",
        "intent": "iot_hue_lightoff",
        "utt": f"lights off in the {room}",
        "annot_utt": f"lights off in the [house_place : {room}]",
    }
    for room in ["kitchen", "hall", "bedroom", "office", "garage", "attic"]
]


def test_convert_external_writes_files_and_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(external, "XLAM_TEST_MULTI", 0)
    monkeypatch.setattr(external, "XLAM_TEST_SINGLE", 0)
    raw, out = tmp_path / "raw", tmp_path / "external"
    answers = [{"name": "getWeatherForecast", "arguments": {"location": "Oslo"}}]
    xlam = [xlam_row("Weather in Oslo?", answers), xlam_row("Sum these", [], (LISTY,))]
    (raw / "xlam").mkdir(parents=True)
    (raw / XLAM_FILE).write_text(json.dumps(xlam))
    write_jsonl(raw / MASSIVE_FILE, LIGHTS_OFF)
    out.mkdir()

    report = convert_external(raw, out)

    assert report["xlam"]["kept"] == 1 and sum(report["xlam"]["reasons"].values()) == 2
    assert report["massive"]["examples"]["train"] == len(LIGHTS_OFF)
    assert json.loads((out / "report.json").read_text()) == report
    assert len(read_jsonl(out / "xlam.jsonl")) == 1 + report["xlam"]["unsupported_variants"]
    assert {row["source"] for row in read_jsonl(out / "massive-train.jsonl")} == {"massive"}


def test_convert_external_with_nothing_downloaded(tmp_path: Path) -> None:
    assert convert_external(tmp_path / "raw", tmp_path) == {}
    assert json.loads((tmp_path / "report.json").read_text()) == {}


def _weather_rows(n_multi: int, n_single: int) -> list[dict[str, str]]:
    def call(city: str) -> dict[str, object]:
        return {"name": "getWeatherForecast", "arguments": {"location": city}}

    multi = [
        xlam_row(
            f"Weather in Town{i}a and also in Town{i}b?", [call(f"Town{i}a"), call(f"Town{i}b")]
        )
        for i in range(n_multi)
    ]
    single = [xlam_row(f"Weather in Village{i}?", [call(f"Village{i}")]) for i in range(n_single)]
    return multi + single


def _convert(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    raw, out = tmp_path / "raw", tmp_path / "external"
    (raw / "xlam").mkdir(parents=True)
    (raw / XLAM_FILE).write_text(json.dumps(rows))
    out.mkdir()
    convert_external(raw, out)
    return out


def test_xlam_test_split_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(external, "XLAM_TEST_MULTI", 4)
    monkeypatch.setattr(external, "XLAM_TEST_SINGLE", 2)
    out = _convert(tmp_path, _weather_rows(10, 10))
    test = read_jsonl(out / "xlam-test.jsonl")
    assert sum(len(r["calls"]) >= 2 for r in test) == 4
    assert sum(len(r["calls"]) == 1 for r in test) == 2
    train = {r["utterance"] for r in read_jsonl(out / "xlam.jsonl")}
    assert not train & {r["utterance"] for r in test}


def test_unsupported_variants_only_from_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(external, "XLAM_TEST_MULTI", 4)
    monkeypatch.setattr(external, "XLAM_TEST_SINGLE", 2)
    monkeypatch.setattr(external, "UNSUPPORTED_SHARE", 1.0)
    out = _convert(tmp_path, _weather_rows(10, 10))
    assert {r["intent"] for r in read_jsonl(out / "xlam-test.jsonl")} == {"call"}


def test_hold_out_leaves_the_shared_rng_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MASSIVE is split with the same rng after xLAM: the hold-out must not change its draws."""
    rows = _weather_rows(10, 10)
    (tmp_path / "x.json").write_text(json.dumps(rows))
    after = []
    for multi, single in ((0, 0), (4, 2)):
        monkeypatch.setattr(external, "XLAM_TEST_MULTI", multi)
        monkeypatch.setattr(external, "XLAM_TEST_SINGLE", single)
        rng = np.random.default_rng(0)
        external._convert_xlam_file(tmp_path / "x.json", tmp_path, rng, seed=0)
        after.append(rng.random())
    assert after[0] == after[1]
