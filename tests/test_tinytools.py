import csv
import gzip
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eve.data.jsonl import write_jsonl
from eve.data.tinytools.build import SOURCES, build_tinytools
from eve.data.tinytools.common import (
    Context,
    EvalSet,
    dialogue,
    humanize,
    number_words,
    plural,
    request_key,
    spell_numbers,
)
from eve.data.tinytools.dialogues import star_constraints
from eve.data.tinytools.homeassistant import Grammar, UnknownName, parse
from eve.data.tinytools.knowledge import everyday_noun, readable
from eve.data.tinytools.problems import svamp


def test_request_key_and_eval_set():
    assert request_key("Please set an alarm for 7.") == request_key("olly set alarm for 8")
    assert request_key("lights") == request_key("2 3") == ""  # too little to count as a match
    evals = EvalSet(frozenset({request_key("turn off the kitchen lights")}))
    assert "Turn off the kitchen lights, please" in evals
    assert "turn off the hall lights" not in evals


def test_context_keep_counts_why():
    evals = EvalSet(frozenset({request_key("what's the weather today")}))
    ctx = Context(np.random.default_rng(0), evals)
    assert ctx.keep(["play some jazz", "louder"])
    assert not ctx.keep("find me a cheap hotel in Paris")
    assert not ctx.keep(["hi", "What's the weather today?"])
    assert ctx.keep("tell me about your trip")  # "trip" stays: only the held-out tools' topics go
    assert ctx.dropped == {"travel topic": 1, "in an eval set": 1}


def test_numbers_in_words():
    assert number_words(21) == "twenty-one" and number_words(305) == "three hundred and five"
    assert number_words(12_040) == "twelve thousand and forty"
    text = "She had 25 apples, paid $5, ran 3.5 km at 7:30 and scored 90%."
    assert spell_numbers(text) == (
        "She had twenty-five apples, paid $5, ran 3.5 km at 7:30 and scored 90%."
    )


def test_helpers():
    assert humanize("HassTurnOn") == "turn on" and humanize("card_arrival") == "card arrival"
    assert humanize("AddToPlaylist") == "add to playlist"
    turns = [
        ("User", "Hi."),
        ("User", "Play jazz."),
        ("", "The assistant searches music"),
        ("Assistant", "Done."),
    ]
    assert (
        dialogue(turns) == "User: Hi. Play jazz.\n(The assistant searches music.)\nAssistant: Done."
    )
    assert star_constraints("[{'Name': 'api.is_equal_to(\"Dr. Johnson\")'}, {'Day': '3'}]") == [
        ("name", "Dr. Johnson"),
        ("day", "3"),
    ]
    assert star_constraints(
        [{"PIN": "7402"}, {"Report": '"someone took $500 from my account and more"'}]
    ) == [("pin", "7402")]  # some dialogues store a list; a long value is left out
    assert [plural(n) for n in ("kettle", "battery", "glass", "child", "toy")] == [
        "kettles",
        "batteries",
        "glasses",
        "children",
        "toys",
    ]
    nouns = frozenset({"kettle", "battery", "glass"})
    assert [everyday_noun(w, nouns) for w in ("Kettles", "batteries", "glasses", "kettl")] == [
        "kettle",
        "battery",
        "glass",
        None,
    ]
    assert readable("The cat sat on the mat. It was warm.")
    assert not readable("Notwithstanding considerable epistemological uncertainties, " * 3)


def test_hassil_templates():
    grammar = Grammar(
        rules={"temp": "(temp|temperature)"},
        lists={
            "brightness": {"range": {"from": 10, "to": 10}},
            "color": {"values": ["red", {"in": "blue[ish]", "out": "blue"}]},
        },
        fills={"area": ["kitchen"]},
    )
    rng = np.random.default_rng(0)
    node = parse("set [the] {area} <temp> to {brightness}[ degrees]")
    for _ in range(20):
        text = " ".join(grammar.sample(node, rng).split())
        assert text in {
            f"set {the}kitchen {t} to 10{d}"
            for the in ("", "the ")
            for t in ("temp", "temperature")
            for d in ("", " degrees")
        }
    colors = {grammar.sample(parse("{color}"), rng) for _ in range(30)}
    assert colors == {"red", "blue", "blueish"}
    assert {grammar.sample(parse("(a;b)"), rng) for _ in range(20)} == {"a b", "b a"}
    with pytest.raises(UnknownName):
        grammar.sample(parse("call {name}"), rng)


def test_svamp_tidies_equations(tmp_path):
    rows = {
        "Body": ["There are 87 oranges ."],
        "Question": ["How many are left ?"],
        "Equation": ["( 87.0 - 2.0 )"],
        "Answer": ["85.0"],
    }
    (tmp_path / "svamp").mkdir()
    pq.write_table(pa.table(rows), tmp_path / "svamp/train.parquet")
    ctx = Context(np.random.default_rng(1))
    docs = list(svamp(tmp_path, ctx))
    assert docs[0].text in {
        "There are 87 oranges. How many are left?\n87 - 2 = 85. The answer is 85.",
        "There are eighty-seven oranges. How many are left?\neighty-seven - two = eighty-five. "
        "The answer is eighty-five.",
    }


@pytest.fixture
def raw(tmp_path):
    """A few sources' raw files, each with one travel example and one eval-set request."""
    root = tmp_path / "raw"
    tm = root / "taskmaster/TM-1-2019"
    tm.mkdir(parents=True)
    utterances = [
        {"speaker": s, "text": t}
        for s, t in [
            ("USER", "I'd like a large pizza."),
            ("ASSISTANT", "Which toppings?"),
            ("USER", "Mushrooms and olives."),
            ("ASSISTANT", "It'll be ready in 20 minutes."),
        ]
    ]
    hotel = [{"speaker": "USER", "text": "Book a hotel near the beach."}] * 4
    convos = [{"utterances": utterances}, {"utterances": hotel}]
    (tm / "self-dialogs.json").write_text(json.dumps(convos))
    (tm / "woz-dialogs.json").write_text("[]")

    hwu = root / "hwu64"
    hwu.mkdir()
    fields = ["scenario", "intent", "status", "answer", "question"]
    with (hwu / "NLU-Data-Home-Domain-Annotated-All.csv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        prompt = "Write what you would tell your PDA in the following scenario: to set an alarm."
        for text in ["wake me up at five", "set an alarm for six am", "alarm at noon please"]:
            writer.writerow(["alarm", "set", "null", text, prompt])
        writer.writerow(["alarm", "set", "IRR", "banana", prompt])
        writer.writerow(["transport", "taxi", "null", "get me a taxi to the station", ""])
        writer.writerow(["alarm", "set", "null", "turn off the kitchen lights", prompt])

    abcd = root / "abcd"
    abcd.mkdir()
    convo = {
        "original": [
            ["customer", "I need to return a shirt."],
            ["agent", "Sure, what is your name?"],
            ["action", "Account has been pulled up for Sam."],
            ["customer", "Sam Lee."],
        ]
    }
    with gzip.open(abcd / "abcd_v1.1.json.gz", "wt") as f:
        json.dump({"train": [convo], "dev": [], "test": []}, f)
    return root


def test_build_tinytools(raw, tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    write_jsonl(
        evals / "val.jsonl",
        [
            {"utterance": "turn off the kitchen lights", "intent": "call"},
            {"utterance": "which toppings", "intent": "none"},  # chitchat isn't matched
        ],
    )
    out = tmp_path / "tinytools"
    names = ["taskmaster-1", "hwu64", "abcd", "clinc150"]
    report = build_tinytools(names, raw, out, tokenizer=None, evals_dir=evals)

    assert report["eval_sets"] == ["val.jsonl"]
    assert report["sources"]["clinc150"] == {"missing": ["clinc150/data_full.json"]}
    tm = pq.read_table(out / "taskmaster-1.parquet").to_pylist()
    assert [row["kind"] for row in tm] == ["dialogue"] and "pizza" in tm[0]["text"]
    assert report["sources"]["taskmaster-1"]["dropped"] == {"travel topic": 1}

    hwu = pq.read_table(out / "hwu64.parquet").column("text").to_pylist()
    text = "\n".join(hwu)
    assert "wake me up at five" in text and "set an alarm" in text
    assert "banana" not in text and "taxi" not in text and "kitchen lights" not in text
    dropped = report["sources"]["hwu64"]["dropped"]
    assert dropped["travel intent"] == dropped["flagged by annotators"] == 1
    assert dropped["in an eval set"] == 1

    abcd = pq.read_table(out / "abcd.parquet").column("text").to_pylist()
    assert abcd == [
        "Customer: I need to return a shirt.\nAgent: Sure, what is your name?\n"
        "(Account has been pulled up for Sam.)\nCustomer: Sam Lee."
    ]
    saved = json.loads((out / "report.json").read_text())
    assert saved["tokens"] == report["tokens"] > 0


def test_every_source_has_downloads():
    for name, source in SOURCES.items():
        assert source.downloads, name
        assert len({d.path for d in source.downloads}) == len(source.downloads), name
