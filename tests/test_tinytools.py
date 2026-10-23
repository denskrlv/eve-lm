import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from eve.cli import main
from eve.data.corpus import iter_documents
from eve.data.generate.tinytools import (
    KINDS,
    CorpusReport,
    Lexicon,
    PromptSpec,
    check_corpus,
    check_document,
    make_prompts,
    near_duplicates,
    word_forms,
    write_parquet,
)
from eve.data.jsonl import read_jsonl, write_jsonl
from eve.data.schema import load_library

ROOT = Path(__file__).parents[1]
LEXICON_DIR = ROOT / "data" / "tinytools"


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return Lexicon.load(LEXICON_DIR)


@pytest.fixture(scope="module")
def heldout():
    return list(load_library(ROOT / "data" / "tools" / "heldout").values())


def spec(kind="scene", noun="lamp", verb="buy", adjective="bright", **kw) -> PromptSpec:
    fields = dict(id="tt-000001", kind=kind, domain="lights", noun=noun, verb=verb,
                  adjective=adjective, feature="", speaker="", prompt="")  # fmt: skip
    return PromptSpec(**(fields | kw))


SCENE = (
    "Nora came home late and the hallway was dark. She had bought a new lamp last week, so she "
    'said to the speaker by the door, "make the hallway brighter, I can\'t see my keys." The '
    "assistant understood that she wanted more light. It turned on the hallway lamp and set it "
    'to the brightest level. Nora found her keys under a pile of letters and laughed. "Thanks," '
    'she said, "you can dim it again now." The assistant lowered the light to a soft glow.'
)


def test_the_word_lists_and_domains_load(lexicon: Lexicon) -> None:
    sizes = (len(lexicon.nouns), len(lexicon.verbs), len(lexicon.adjectives))
    assert sum(sizes) > 1200 and min(sizes) > 200
    assert lexicon.verbs["buy"] == ("bought",)
    assert len(lexicon.domains) > 90
    concept = {d.name for d in lexicon.domains if d.concept_only}
    assert {"arithmetic", "measurements and units", "flights"} <= concept
    assert len(lexicon.names) > 250 and "Priya" in lexicon.names


def test_prompts_are_seeded_and_follow_the_mix(lexicon: Lexicon) -> None:
    specs = make_prompts(lexicon, 2000, seed=0)
    assert [s.to_json() for s in specs[:50]] == [s.to_json() for s in make_prompts(lexicon, 50, 0)]
    assert specs[0] != make_prompts(lexicon, 1, seed=1)[0]
    assert len({s.id for s in specs}) == 2000
    counts = Counter(s.kind for s in specs)
    for kind, share in KINDS.items():
        assert counts[kind] / 2000 == pytest.approx(share, abs=0.04)
    concept = {d.name for d in lexicon.domains if d.concept_only}
    assert not any(s.kind == "card" and s.domain in concept for s in specs)
    assert any(s.kind != "card" and s.domain in concept for s in specs)
    for s in specs[:200]:
        assert f'"{s.noun}"' in s.prompt and f'"{s.verb}"' in s.prompt
        assert bool(s.feature) == (s.kind in ("scene", "ways", "dialogue"))
        assert bool(s.speaker) == (s.kind in ("scene", "dialogue"))
        assert PromptSpec.from_json(json.loads(json.dumps(s.to_json()))) == s


def test_prompts_keep_the_pilot_draws_and_vary_the_wording(lexicon: Lexicon) -> None:
    pilot = list(read_jsonl(LEXICON_DIR / "pilot" / "prompts.jsonl"))[:100]
    specs = make_prompts(lexicon, 500, seed=0)
    same = ("id", "kind", "domain", "noun", "verb", "adjective", "speaker")
    for old, new in zip(pilot, specs, strict=False):
        assert {k: old[k] for k in same} == {k: getattr(new, k) for k in same}
    times = {s.feature for s in specs if s.feature.startswith("a time or date phrase")}
    assert len(times) > 3
    for s in specs:
        if s.kind == "scene":
            name = s.prompt.split("The person's name is ")[1].split(".")[0]
            assert name in lexicon.names and "understood" not in s.prompt
        if s.kind == "card":
            assert "does: or inputs:" in s.prompt
        assert "Spell these three words correctly." in s.prompt


def test_word_forms_cover_inflections() -> None:
    assert {"stopped", "stopping", "stops"} <= word_forms("stop")
    assert {"berries"} <= word_forms("berry") and {"bigger", "biggest"} <= word_forms("big")
    assert {"baking", "baked"} <= word_forms("bake")
    assert "bought" in word_forms("buy", ("bought",))
    assert {"shelves"} <= word_forms("shelf") and {"knives"} <= word_forms("knife")
    assert "automatically" in word_forms("automatic")
    assert "foggy" in word_forms("fog") and "salty" in word_forms("salt")


def test_a_good_scene_passes(lexicon: Lexicon, heldout) -> None:
    assert check_document(spec(), SCENE, lexicon, heldout) == []
    possessive = SCENE.replace("new lamp", "new reading light").replace("hallway lamp", "lamp's")
    assert "lamp " not in possessive.replace("lamp's", "")
    assert check_document(spec(), possessive, lexicon, heldout) == []


@pytest.mark.parametrize(
    ("change", "text", "reason"),
    [
        ({"noun": "piano"}, SCENE, "doesn't use 'piano'"),
        ({}, "The lamp was bright. She bought it.", "words; a scene needs"),
        ({}, "Sure, here is a story. " + SCENE, "starts with a reply"),
        ({}, SCENE + " She typed <tool> by mistake.", "special token <tool>"),
        ({}, SCENE + " Then it ran convert_units for her.", "held-out tool convert_units"),
        ({}, SCENE + " It called add(20, 10) as well.", "held-out tool add"),
        ({"kind": "dialogue"}, SCENE, "a dialogue needs 4+ turns"),
        ({"kind": "card"}, SCENE, "a tool card needs a line 'name:"),
    ],
)
def test_a_bad_document_is_rejected(lexicon, heldout, change, text, reason) -> None:
    errors = check_document(replace(spec(), **change), text, lexicon, heldout)
    assert any(reason in error for error in errors), errors


def test_a_card_and_a_dialogue_pass(lexicon: Lexicon, heldout) -> None:
    card = (
        "name: set_porch_lamp\n"
        "does: turns the porch lamp on or off and makes it brighter or dimmer\n"
        "inputs: on or off, and how bright, from 1 to 10\n"
        "people say:\n"
        "- can you put the light on outside, the front step is dark\n"
        "- make the porch brighter, my friends just bought snacks and are coming over\n"
        "- switch off the lamp by the front door\n"
        "- the outside light is too strong, take it down a bit"
    )
    assert check_document(spec(kind="card"), card, lexicon, heldout) == []
    dialogue = (
        "Person: I just bought a new lamp, can you turn it on?\n"
        "Assistant: Sure. Which room is the lamp in?\n"
        "Person: The bedroom, next to the bed.\n"
        "Assistant: Done. The bedroom lamp is on. Do you want it bright or soft?\n"
        "Person: Bright, please, I want to read."
    )
    assert check_document(spec(kind="dialogue"), dialogue, lexicon, heldout) == []
    math = card.replace("set_porch_lamp", "add").replace("turns the porch", "Add two numbers or")
    assert any(
        "held-out tool add" in e for e in check_document(spec(kind="card"), math, lexicon, heldout)
    )
    assert (
        check_document(spec(), SCENE + " Then she asked it to add two numbers.", lexicon, heldout)
        == []
    )


def test_near_duplicates_point_at_the_earlier_document() -> None:
    other = "A quiet garden needs water every morning when the weather is dry and warm outside."
    copy = SCENE.replace("Nora", "Lena")
    assert near_duplicates({"a": SCENE, "b": other, "c": copy}) == {"c": "a"}


def test_check_corpus_keeps_the_good_ones(lexicon: Lexicon, heldout, tmp_path: Path) -> None:
    specs = [spec(id="tt-1"), spec(id="tt-2", noun="piano"), spec(id="tt-3"), spec(id="tt-4")]
    docs = {"tt-1": SCENE, "tt-2": SCENE, "tt-3": SCENE.replace("Nora", "Lena"), "tt-9": "x"}
    report = check_corpus(specs, docs, lexicon, heldout)
    assert [s.id for s, _ in report.accepted] == ["tt-1"]
    assert set(report.rejected) == {"tt-2", "tt-3"} and report.missing == ["tt-4"]
    assert report.unknown == ["tt-9"]
    stats = report.stats(lexicon)
    assert stats["accepted"] == 1 and stats["prompts"] == 4 and stats["missing"] == 1
    assert stats["reject_reasons"] == {"doesn't use …": 1, "nearly repeats tt-…": 1}
    assert stats["top_names"] == {"Nora": 1.0}
    assert list(stats["repeated_phrases"]["scene"].values()) == [1.0] * 5  # one document

    write_parquet(report.accepted, tmp_path / "out.parquet")
    assert list(iter_documents([tmp_path / "out.parquet"])) == [SCENE]


def test_stats_report_copied_examples(lexicon: Lexicon) -> None:
    time = 'a time or date phrase, like "tomorrow at seven"'
    texts = [
        SCENE + " Remind me tomorrow at seven.",
        SCENE.replace("Nora", "Ada") + " Tomorrow, at seven!",
    ]
    docs = [(spec(id=f"tt-{i}", feature=time), text) for i, text in enumerate(texts)]
    docs.append((spec(id="tt-9", feature='a time or date phrase, like "in an hour"'), SCENE))
    copied = CorpusReport(accepted=docs).stats(lexicon)["copied_examples"]
    assert copied == {'a time or date phrase, like "…"': pytest.approx(2 / 3, abs=0.001)}
    number = spec(feature="a number written in words, like three hundred")
    copied = CorpusReport(accepted=[(number, SCENE + " Three hundred.")]).stats(lexicon)
    assert copied["copied_examples"] == {"a number written in words, like …": 1.0}


def test_eve_corpus_prompts_and_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "pilot"
    main(["corpus", "prompts", "--n", "20", "--out", str(run)])
    assert "20 prompts" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        main(["corpus", "prompts", "--n", "20", "--out", str(run)])

    specs = [PromptSpec.from_json(row) for row in read_jsonl(run / "prompts.jsonl")]
    write_jsonl(run / "documents.jsonl", [{"id": specs[0].id, "text": SCENE}])
    main(["corpus", "check", str(run), "--json"])
    stats = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert stats["prompts"] == 20 and stats["missing"] == 19
    assert (run / "tinytools.parquet").exists()
    assert set(json.loads((run / "report.json").read_text())) == {"stats", "rejected", "missing"}

    main(["corpus", "check", str(run)])
    assert "eve corpus check" in capsys.readouterr().out
