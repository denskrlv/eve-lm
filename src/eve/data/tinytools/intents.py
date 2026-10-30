"""Intent datasets ("ways to say it", tool cards, scenes): HWU64, CLINC150, Banking77 and SNIPS,
plus TaPaCo's paraphrase sets."""

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from eve.data.draw import pick
from eve.data.tinytools.common import Context, Doc, Intent, chunks, clean, humanize, intent_docs

# HWU64's travel intents: transport_* and qa_currency, as in fine-tuning; datetime_convert
# (time zones) too, which MASSIVE's conversion leaves out with them.
HWU_TRAVEL = ("transport_", "qa_currency", "datetime_convert")
HWU_KEPT_STATUS = ("", "null")  # other statuses mark answers the annotators flagged
HWU_MIXED = ("general_quirky",)  # anything that fits no intent: not one request reworded
_HWU_SCENARIO = re.compile(r"scenario:\s*(?:to\s+)?(.+?)\.?\s*$", re.IGNORECASE)
CLINC_TRAVEL = frozenset(
    "book_flight book_hotel flight_status exchange_rate international_visa travel_alert "
    "travel_notification travel_suggestion lost_luggage carry_on plug_type vaccines "
    "car_rental".split()
)
BANKING_TRAVEL = re.compile(r"exchange|currenc|fiat")  # Banking77 categories left out
TAPACO_MIN_SET = 3  # paraphrase sets with fewer sentences are left out
TAPACO_SETS_LOW, TAPACO_SETS_HIGH = 1, 4  # paraphrase sets per document


def hwu_about(question: str, label: str) -> tuple[str, bool]:
    """What an HWU64 intent asks for, from the annotators' prompt ("...in the following
    scenario: to set an alarm."), and whether it is a verb phrase."""
    if match := _HWU_SCENARIO.search(question):
        return match.group(1).strip(), True
    return humanize(label), False


def hwu64(raw: Path, ctx: Context) -> Iterator[Doc]:
    """HWU64 (NLU-Evaluation-Data), grouped by scenario and intent."""
    groups: dict[str, list[str]] = {}
    abouts: dict[str, tuple[str, bool]] = {}
    path = raw / "hwu64/NLU-Data-Home-Domain-Annotated-All.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            label = f"{row['scenario']}_{row['intent']}"
            if label.startswith(HWU_TRAVEL):
                ctx.drop("travel intent")
            elif label in HWU_MIXED:
                ctx.drop("mixed intent")
            elif (row.get("status") or "").strip() not in HWU_KEPT_STATUS:
                ctx.drop("flagged by annotators")
            else:
                groups.setdefault(label, []).append(row["answer"])
                abouts.setdefault(label, hwu_about(row.get("question") or "", label))
    intents = (
        Intent(label, abouts[label][0], tuple(requests), verb=abouts[label][1])
        for label, requests in sorted(groups.items())
    )
    return intent_docs(intents, ctx)


def clinc150(raw: Path, ctx: Context) -> Iterator[Doc]:
    """CLINC150's in-scope requests from every split, without its travel intents."""
    data = json.loads((raw / "clinc150/data_full.json").read_text())
    groups: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        for text, label in data[split]:
            if label in CLINC_TRAVEL:
                ctx.drop("travel intent")
            else:
                groups.setdefault(label, []).append(text)
    intents = (Intent(k, humanize(k), tuple(v)) for k, v in sorted(groups.items()))
    return intent_docs(intents, ctx)


def banking77(raw: Path, ctx: Context) -> Iterator[Doc]:
    """Banking77's train and test requests, without the currency-exchange categories."""
    groups: dict[str, list[str]] = {}
    for split in ("train", "test"):
        with (raw / f"banking77/{split}.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if BANKING_TRAVEL.search(row["category"]):
                    ctx.drop("travel intent")
                else:
                    groups.setdefault(row["category"], []).append(row["text"])
    intents = (Intent(k, humanize(k), tuple(v)) for k, v in sorted(groups.items()))
    return intent_docs(intents, ctx)


def _read_text(path: Path) -> str:
    """Some SNIPS files aren't valid UTF-8; those are Latin-1."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def snips(raw: Path, ctx: Context) -> Iterator[Doc]:
    """SNIPS's seven intents (the 2017 custom-intent-engines training files)."""
    groups: dict[str, list[str]] = {}
    for path in sorted((raw / "snips").glob("train_*_full.json")):
        for label, examples in json.loads(_read_text(path)).items():
            for example in examples:
                groups.setdefault(label, []).append("".join(p["text"] for p in example["data"]))
    intents = (Intent(k, humanize(k), tuple(v)) for k, v in sorted(groups.items()))
    return intent_docs(intents, ctx)


def tapaco(raw: Path, ctx: Context) -> Iterator[Doc]:
    """TaPaCo's English paraphrase sets of three or more sentences, a few sets per document."""
    table = pq.read_table(raw / "tapaco/en.parquet", columns=["paraphrase_set_id", "paraphrase"])
    sets: dict[str, list[str]] = {}
    for set_id, text in zip(*(table.column(i).to_pylist() for i in range(2)), strict=True):
        sets.setdefault(set_id, []).append(clean(text))
    kept = []
    for sentences in sets.values():
        sentences = list(dict.fromkeys(sentences))
        if len(sentences) < TAPACO_MIN_SET:
            ctx.drop("fewer than 3 paraphrases")
        elif ctx.keep(sentences):
            kept.append(sentences)
    order = ctx.rng.permutation(len(kept))
    for group in chunks([kept[i] for i in order], ctx.rng, TAPACO_SETS_LOW, TAPACO_SETS_HIGH):
        parts = []
        for sentences in group:
            header = pick(
                ctx.rng,
                ("These mean the same:", "Other ways to say it:", "Same meaning, other words:"),
            )
            parts.append("\n".join([header, *sentences]))
        yield Doc("\n\n".join(parts), "ways")
