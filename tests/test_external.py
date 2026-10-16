import json

import numpy as np

from colibri.data.generate.external import (
    HELD_OUT,
    MASSIVE_TOOLS,
    convert_massive,
    convert_xlam,
    short,
    snake,
    unsupported_variant,
    xlam_tool,
)
from colibri.data.render import TaskExample

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
    assert convert_xlam(xlam_row("add 1 and 2", sums, [LISTY]), rng)[1] == "held-out topic"


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
