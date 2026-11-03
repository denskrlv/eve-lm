"""Each TinyTools converter on a few rows of its source's raw format: what it keeps, and what it
drops and why (``Context.dropped``). The documents are shuffled and worded at random, so the
checks are on kinds, phrases and counts, not whole texts. ``fetch`` runs with the network
stubbed out."""

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eve.data.tinytools import dialogues, fetch, homeassistant, intents, knowledge, problems
from eve.data.tinytools.common import Context, Doc

NOUNS = frozenset({"kettle", "cat", "dog", "door", "teacher"})


def context() -> Context:
    return Context(np.random.default_rng(0), names=("Sam",), nouns=NOUNS, person_nouns={"teacher"})


def text_of(docs: list[Doc]) -> str:
    return "\n".join(doc.text for doc in docs)


def write_parquet(path: Path, columns: dict[str, list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# ---- how things work -------------------------------------------------------------------------


def test_genericskb_keeps_written_facts_about_everyday_things(tmp_path: Path) -> None:
    rows = [
        ("Waterloo", "kettle", "Kettles boil water quickly.", 0.9),
        ("ConceptNet", "kettle", "Kettles has-part handles.", 0.9),
        ("Waterloo", "kettle", "Kettles are odd.", 0.2),
        ("Waterloo", "teacher", "Teachers grade homework.", 0.9),
        ("Waterloo", "kettle", " ".join(["Kettles"] + ["boil"] * 30) + ".", 0.9),
        ("Waterloo", "kettle", "Kettles in hotels are small.", 0.9),
    ]
    columns = ["source", "term", "generic_sentence", "score"]
    write_parquet(
        tmp_path / "genericskb/best.parquet",
        {c: [row[i] for row in rows] for i, c in enumerate(columns)},
    )
    ctx = context()

    docs = list(knowledge.genericskb(tmp_path, ctx))

    assert {doc.kind for doc in docs} == {"how"}
    assert "Kettles boil water quickly." in text_of(docs)
    assert ctx.dropped == {
        "synthesized from triples": 1,
        "low quality score": 1,
        "not about an everyday thing": 1,
        "long sentence": 1,
        "travel topic": 1,
    }


def test_atomic_writes_object_uses_and_reasons(tmp_path: Path) -> None:
    rows = [
        ("kettle", "ObjectUse", "boil water"),
        ("kettle", "CapableOf", "whistle"),
        ("rocket", "ObjectUse", "fly to the moon"),
        ("PersonX puts on a coat", "xIntent", "to stay warm"),
        ("PersonX wins the race", "xIntent", "successful"),
        ("a row", "with two fields"),
    ]
    folder = tmp_path / "atomic2020"
    folder.mkdir()
    for split in ("train", "dev", "test"):
        with (folder / f"{split}.tsv").open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows if split == "train" else [])
    ctx = context()

    docs = list(knowledge.atomic(tmp_path, ctx))

    text = text_of(docs)
    assert {doc.kind for doc in docs} == {"how"}
    assert "kettle" in text and "boil water" in text and "whistle" in text
    assert "stay warm" in text and "Sam" in text and "rocket" not in text
    assert ctx.dropped == {"not about an everyday thing": 1, "reason not a to-phrase": 1}


STORY = "The cat sat by the kettle. A dog ran to the door. The cat was happy."


def test_cosmopedia_keeps_plain_stories_for_young_children(tmp_path: Path) -> None:
    hard = "The cat, dog and kettle contemplated extraordinarily sophisticated considerations."
    rows = [
        (STORY, "young_children"),
        (STORY.replace("happy", "sleepy"), "adults"),
        ("The cat slept.", "young_children"),
        (hard, "young_children"),
        ("The cat and the dog took a flight with the kettle.", "young_children"),
    ]
    columns = {"text": [r[0] for r in rows], "audience": [r[1] for r in rows]}
    write_parquet(tmp_path / "cosmopedia/stories/0.parquet", columns)
    write_parquet(tmp_path / "cosmopedia/wikihow/0.parquet", columns)
    ctx = context()

    stories = list(knowledge.cosmopedia_stories(tmp_path, ctx))
    wikihow = list(knowledge.cosmopedia_wikihow(tmp_path, context()))

    assert stories == [Doc(STORY, "how")]
    assert ctx.dropped == {
        "audience not young_children": 1,
        "few everyday nouns": 1,
        "hard to read": 1,
        "travel topic": 1,
    }
    assert [doc.text for doc in wikihow] == [STORY, STORY.replace("happy", "sleepy")]


# ---- intent datasets -------------------------------------------------------------------------


def test_clinc150_drops_travel_intents(tmp_path: Path) -> None:
    data = {
        "train": [["wake me up at seven", "alarm"], ["book me a flight to rome", "book_flight"]],
        "val": [["set an alarm for noon", "alarm"]],
        "test": [],
        "oos_train": [["what is love", "oos"]],
    }
    write_json(tmp_path / "clinc150/data_full.json", data)
    ctx = context()

    text = text_of(list(intents.clinc150(tmp_path, ctx)))

    assert "wake me up at seven" in text and "set an alarm for noon" in text
    assert "rome" not in text and "what is love" not in text
    assert ctx.dropped == {"travel intent": 1}


def test_banking77_drops_currency_categories(tmp_path: Path) -> None:
    folder = tmp_path / "banking77"
    folder.mkdir()
    rows = {
        "train": [("when will my card arrive", "card_arrival")],
        "test": [("what is the exchange rate today", "exchange_rate")],
    }
    for split, items in rows.items():
        with (folder / f"{split}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "category"])
            writer.writerows(items)
    ctx = context()

    text = text_of(list(intents.banking77(tmp_path, ctx)))

    assert "when will my card arrive" in text and "card arrival" in text
    assert "exchange rate today" not in text
    assert ctx.dropped == {"travel intent": 1}


def test_snips_reads_latin1_files(tmp_path: Path) -> None:
    folder = tmp_path / "snips"
    folder.mkdir()
    play = {"PlayMusic": [{"data": [{"text": "play some "}, {"text": "jazz"}]}]}
    write_json(folder / "train_PlayMusic_full.json", play)
    rate = {"RateBook": [{"data": [{"text": "give the café book five stars"}]}]}
    (folder / "train_RateBook_full.json").write_bytes(
        json.dumps(rate, ensure_ascii=False).encode("latin-1")
    )

    text = text_of(list(intents.snips(tmp_path, context())))

    assert "play some jazz" in text and "give the café book five stars" in text


def test_tapaco_needs_three_paraphrases(tmp_path: Path) -> None:
    sets = [
        ("1", "I am hungry."),
        ("1", "I'm  hungry."),
        ("1", "I feel hungry."),
        ("1", "I want food."),
        ("2", "It is late."),
        ("2", "It's late."),
    ]
    columns = {"paraphrase_set_id": [s for s, _ in sets], "paraphrase": [p for _, p in sets]}
    write_parquet(tmp_path / "tapaco/en.parquet", columns)
    ctx = context()

    docs = list(intents.tapaco(tmp_path, ctx))

    assert [doc.kind for doc in docs] == ["ways"]
    assert "I'm hungry." in docs[0].text and "late" not in docs[0].text  # whitespace cleaned
    assert ctx.dropped == {"fewer than 3 paraphrases": 1}


# ---- dialogues -------------------------------------------------------------------------------

PIZZA = [
    ("USER", "I'd like a large pizza."),
    ("ASSISTANT", "Which toppings?"),
    ("USER", "Mushrooms and olives."),
    ("ASSISTANT", "It'll be ready in 20 minutes."),
]


def convo(turns: list[tuple[str, str]]) -> dict[str, object]:
    return {"utterances": [{"speaker": s, "text": t} for s, t in turns]}


def test_taskmaster_2_skips_travel_files(tmp_path: Path) -> None:
    folder = tmp_path / "taskmaster/TM-2-2020/data"
    write_json(folder / "flights.json", [convo(PIZZA)])
    write_json(folder / "restaurant-search.json", [convo(PIZZA)])
    ctx = context()

    docs = list(dialogues.taskmaster_2(tmp_path, ctx))

    assert [doc.kind for doc in docs] == ["dialogue"]
    assert docs[0].text.startswith("User: I'd like a large pizza.\nAssistant: Which toppings?")
    assert ctx.dropped == {"travel file": 1}


def test_taskmaster_3_reads_its_data_files(tmp_path: Path) -> None:
    folder = tmp_path / "taskmaster/TM-3-2020/data"
    write_json(folder / "data_00.json", [convo(PIZZA)])
    write_json(folder / "notes.json", [convo(PIZZA)])  # not a data_*.json file

    docs = list(dialogues.taskmaster_3(tmp_path, context()))

    assert [doc.kind for doc in docs] == ["dialogue"]


def star_dialogue(domain: str, task: str, events: list[dict[str, str]]) -> dict[str, object]:
    return {
        "Scenario": {"Domains": [domain], "WizardCapabilities": [{"Task": task}]},
        "Events": events,
    }


def test_star_dialogues_and_api_cards(tmp_path: Path) -> None:
    def said(agent: str, text: str) -> dict[str, str]:
        return {"Agent": agent, "Action": "utter", "Text": text}

    query = {
        "Agent": "Wizard",
        "Action": "query",
        "APIName": "bank_balance",
        "Constraints": "[{'Name': 'api.is_equal_to(\"Sam Lee\")'}]",
    }
    bank = [
        said("User", "I want to check my bank balance please"),
        said("Wizard", "Sure, what is your name?"),
        said("User", "Sam Lee"),
        query,
        said("Wizard", "Your balance is 20 dollars."),
    ]
    folder = tmp_path / "star/dialogues"
    write_json(folder / "1.json", star_dialogue("bank", "bank_balance", bank))
    write_json(folder / "2.json", star_dialogue("hotel", "hotel_book", bank))
    write_json(folder / "10.json", star_dialogue("bank", "bank_balance", bank[:2]))
    api = {"input": [{"Name": "Name", "ReadableName": "Name"}], "required": ["Name"]}
    write_json(tmp_path / "star/apis/bank_balance.json", api)
    ctx = context()

    docs = list(dialogues.star(tmp_path, ctx))

    assert [doc.kind for doc in docs] == ["dialogue", "card"]
    assert "(The assistant searches bank balance for name Sam Lee.)" in docs[0].text
    assert "bank_balance" in docs[1].text and "Name, required" in docs[1].text
    assert "I want to check my bank balance please" in docs[1].text
    assert ctx.dropped == {"travel domain": 1, "too short": 1}


# ---- Home Assistant --------------------------------------------------------------------------


def write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_home_assistant_samples_templates_and_test_cases(tmp_path: Path) -> None:
    root = tmp_path / "home-assistant"
    write_yaml(root / "rules/en/_common.yaml", "expansion_rules:\n  switch: (turn|switch)\n")
    write_yaml(root / "lists/en/_common.yaml", "lists:\n  color:\n    values: [red, blue]\n")
    sentences = (
        "language: en\n"
        "data:\n"
        "  - sentences:\n"
        '      - "<switch> on [the] {name}"\n'
        '      - "make the {name} {color}"\n'
        '      - "set it to {unknown}"\n'
    )
    write_yaml(root / "sentences/en/HassTurnOn/x.yaml", sentences)
    tests = (
        "language: en\n"
        "entities:\n"
        "  - name: kitchen light\n"
        "tests:\n"
        "  - sentences: [turn on the kitchen light]\n"
        "    intent: {name: HassTurnOn}\n"
        "    response: Turned on the light\n"
    )
    write_yaml(root / "tests/en/HassTurnOn/x.yaml", tests)
    ctx = context()

    docs = list(homeassistant.home_assistant(tmp_path, ctx))

    text = text_of(docs)
    assert "turn on" in text and "kitchen light" in text
    assert "make the kitchen light red" in text or "make the kitchen light blue" in text
    assert [doc for doc in docs if doc.kind == "scene" and "Turned on the light" in doc.text]
    assert ctx.dropped == {"template with an unknown list or rule": 1}


# ---- word problems ---------------------------------------------------------------------------


def test_gsm8k_strips_calculator_notes_and_states_the_answer(tmp_path: Path) -> None:
    question = "Sam has 2 bags with 3 apples each. How many apples does Sam have?"
    answer = "Each bag has 3, so 2*3=<<2*3=6>>6 apples.\n#### 6.0"
    write_parquet(
        tmp_path / "gsm8k/train.parquet", {"question": [question] * 2, "answer": [answer] * 2}
    )
    ctx = context()

    docs = list(problems.gsm8k(tmp_path, ctx))

    assert len(docs) == 1 and "<<" not in docs[0].text and "####" not in docs[0].text
    assert docs[0].text.endswith(("The answer is 6.", "The answer is six."))
    assert ctx.dropped == {"duplicate question": 1}


# ---- downloads -------------------------------------------------------------------------------


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """What ``fetch._get`` returns per URL: no download leaves the machine."""
    served: dict[str, bytes] = {}
    monkeypatch.setattr(fetch, "_get", lambda url: served[url])
    return served


def test_fetch_saves_files_once(tmp_path: Path, downloads: dict[str, bytes]) -> None:
    downloads["https://example.org/a.txt"] = b"hello"
    file = fetch.Url("https://example.org/a.txt", "a/a.txt")

    assert fetch.fetch((file,), tmp_path) == ["a/a.txt"]
    assert (tmp_path / "a/a.txt").read_bytes() == b"hello"
    downloads["https://example.org/a.txt"] = b"again"
    assert fetch.fetch((file,), tmp_path) == []
    assert fetch.fetch((file,), tmp_path, force=True) == ["a/a.txt"]
    assert (tmp_path / "a/a.txt").read_bytes() == b"again"


def zipped(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def tarred(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("url", "pack"), [("https://x.org/a.zip", zipped), ("https://x.org/a.tar.gz", tarred)]
)
def test_fetch_extracts_only_the_wanted_archive_members(
    tmp_path: Path, downloads: dict[str, bytes], url: str, pack: object
) -> None:
    members = {"top/keep/one.txt": b"1", "top/keep/sub/two.txt": b"2", "top/other/three.txt": b"3"}
    downloads[url] = pack(members)  # type: ignore[operator]

    saved = fetch.fetch((fetch.Archive(url, ("keep/",), "arch"),), tmp_path)

    assert saved == ["arch"]
    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["arch/keep/one.txt", "arch/keep/sub/two.txt"]


def test_fetch_refuses_archive_members_outside_the_folder(
    tmp_path: Path, downloads: dict[str, bytes]
) -> None:
    downloads["https://x.org/a.zip"] = zipped({"top/keep/../../../evil.txt": b"!"})
    with pytest.raises(ValueError, match="outside the folder"):
        fetch.fetch((fetch.Archive("https://x.org/a.zip", ("keep/",), "arch"),), tmp_path)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_fetch_reports_an_archive_without_the_wanted_members(
    tmp_path: Path, downloads: dict[str, bytes]
) -> None:
    downloads["https://x.org/a.zip"] = zipped({"top/other/one.txt": b"1"})
    with pytest.raises(ValueError, match="nothing under"):
        fetch.fetch((fetch.Archive("https://x.org/a.zip", ("keep/",), "arch"),), tmp_path)


def test_fetch_moves_a_hub_file_into_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def download(repo: str, filename: str, repo_type: str, local_dir: Path) -> str:
        assert (repo, repo_type) == ("org/data", "dataset")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True)
        path.write_bytes(b"rows")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    saved = fetch.fetch(
        (fetch.Hub("org/data", "data/train.parquet", "hub/train.parquet"),), tmp_path
    )

    assert saved == ["hub/train.parquet"]
    assert (tmp_path / "hub/train.parquet").read_bytes() == b"rows"
    assert not (tmp_path / "hub/.hub").exists()
