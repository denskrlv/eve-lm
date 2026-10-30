"""The TinyTools round-1 sources (md/tinytools-datasets.md): where each comes from, how many
tokens it may give, and the build that writes one parquet file per source."""

import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eve import defaults
from eve.data.draw import shuffled
from eve.data.tinytools import dialogues, homeassistant, intents, knowledge, problems
from eve.data.tinytools.common import TRAVEL, Context, Doc, EvalSet, read_sections, read_words
from eve.data.tinytools.fetch import Archive, Download, Hub, Url
from eve.tokenizer.bpe import BPETokenizer

BYTES_PER_TOKEN = 4.0  # a rough count when there is no tokenizer to count with
MIN_DOC_WORDS = 5

GITHUB = "https://raw.githubusercontent.com"
TM = f"{GITHUB}/google-research-datasets/Taskmaster/master"
SNIPS = f"{GITHUB}/sonos/nlu-benchmark/master/2017-06-custom-intent-engines"
SNIPS_INTENTS = (
    "AddToPlaylist BookRestaurant GetWeather PlayMusic RateBook SearchCreativeWork "
    "SearchScreeningEvent"
).split()
ATOMIC_ZIP = (
    "https://drive.usercontent.google.com/download?id=1uuY0Y_s8dhxdsoOe8OHRgsqf-9qJIai7"
    "&export=download&confirm=t"
)
COSMO = "HuggingFaceTB/cosmopedia"

type Converter = Callable[[Path, Context], Iterable[Doc]]


@dataclass(frozen=True)
class Source:
    """A round-1 source: its converter, its downloads and its cap in tokens (None: all)."""

    convert: Converter
    downloads: tuple[Download, ...]
    licence: str
    cap: int | None = None


SOURCES: dict[str, Source] = {
    "taskmaster-1": Source(
        dialogues.taskmaster_1,
        tuple(
            Url(f"{TM}/TM-1-2019/{f}", f"taskmaster/TM-1-2019/{f}")
            for f in ("self-dialogs.json", "woz-dialogs.json")
        ),
        "CC BY 4.0",
    ),
    "taskmaster-2": Source(
        dialogues.taskmaster_2,
        tuple(  # flights.json and hotels.json are travel
            Url(f"{TM}/TM-2-2020/data/{f}.json", f"taskmaster/TM-2-2020/data/{f}.json")
            for f in ("food-ordering", "movies", "music", "restaurant-search", "sports")
        ),
        "CC BY 4.0",
    ),
    "taskmaster-3": Source(
        dialogues.taskmaster_3,
        tuple(
            Url(f"{TM}/TM-3-2020/data/{f}", f"taskmaster/TM-3-2020/data/{f}")
            for f in ("data_00.json", "data_01.json")
        ),
        "CC BY 4.0",
        cap=1_000_000,  # movie tickets only
    ),
    "star": Source(
        dialogues.star,
        (
            Archive(
                "https://codeload.github.com/RasaHQ/STAR/tar.gz/refs/heads/master",
                ("dialogues/", "apis/apis/"),
                "star",
            ),
        ),
        "MIT",
    ),
    "abcd": Source(
        dialogues.abcd,
        (
            Url(
                f"{GITHUB}/asappresearch/abcd/master/data/abcd_v1.1.json.gz",
                "abcd/abcd_v1.1.json.gz",
            ),
        ),
        "MIT",
    ),
    "hwu64": Source(
        intents.hwu64,
        (
            Url(
                f"{GITHUB}/xliuhw/NLU-Evaluation-Data/master/AnnotatedData/"
                "NLU-Data-Home-Domain-Annotated-All.csv",
                "hwu64/NLU-Data-Home-Domain-Annotated-All.csv",
            ),
        ),
        "CC BY 4.0",
    ),
    "home-assistant": Source(
        homeassistant.home_assistant,
        (
            Archive(
                "https://codeload.github.com/home-assistant/intents/tar.gz/refs/heads/main",
                ("sentences/en/", "tests/en/", "rules/en/", "lists/", "LICENSE"),
                "home-assistant",
            ),
        ),
        "CC BY 4.0",
    ),
    "clinc150": Source(
        intents.clinc150,
        (Url(f"{GITHUB}/clinc/oos-eval/master/data/data_full.json", "clinc150/data_full.json"),),
        "CC BY 3.0",
    ),
    "banking77": Source(
        intents.banking77,
        tuple(
            Url(
                f"{GITHUB}/PolyAI-LDN/task-specific-datasets/master/banking_data/{s}.csv",
                f"banking77/{s}.csv",
            )
            for s in ("train", "test")
        ),
        "CC BY 4.0",
        cap=200_000,  # one domain
    ),
    "snips": Source(
        intents.snips,
        tuple(
            Url(f"{SNIPS}/{i}/train_{i}_full.json", f"snips/train_{i}_full.json")
            for i in SNIPS_INTENTS
        ),
        "CC0 1.0",
    ),
    "tapaco": Source(
        intents.tapaco,
        (Hub("community-datasets/tapaco", "en/train-00000-of-00001.parquet", "tapaco/en.parquet"),),
        "CC BY 2.0",
    ),
    "genericskb": Source(
        knowledge.genericskb,
        (
            Hub(
                "community-datasets/generics_kb",
                "generics_kb_best/train-00000-of-00001.parquet",
                "genericskb/best.parquet",
            ),
        ),
        "CC BY 4.0",
        cap=3_000_000,
    ),
    "atomic2020": Source(
        knowledge.atomic,
        (Archive(ATOMIC_ZIP, ("train.tsv", "dev.tsv", "test.tsv"), "atomic2020"),),
        "CC BY 4.0",
        cap=2_000_000,
    ),
    "cosmopedia-stories": Source(
        knowledge.cosmopedia_stories,
        (
            Hub(
                COSMO,
                "data/stories/train-00000-of-00043.parquet",
                "cosmopedia/stories/train-00000-of-00043.parquet",
            ),
        ),
        "Apache 2.0",
        cap=5_000_000,
    ),
    "cosmopedia-wikihow": Source(
        knowledge.cosmopedia_wikihow,
        tuple(
            Hub(
                COSMO,
                f"data/wikihow/train-0000{i}-of-00002.parquet",
                f"cosmopedia/wikihow/train-0000{i}-of-00002.parquet",
            )
            for i in range(2)
        ),
        "Apache 2.0",
        cap=5_000_000,
    ),
    "gsm8k": Source(
        problems.gsm8k,
        (Hub("openai/gsm8k", "main/train-00000-of-00001.parquet", "gsm8k/train.parquet"),),
        "MIT",
    ),
    "svamp": Source(
        problems.svamp,
        tuple(
            Hub("ChilleD/SVAMP", f"data/{s}-00000-of-00001.parquet", f"svamp/{s}.parquet")
            for s in ("train", "test")
        ),
        "MIT",
    ),
}  # MAWPS is left out: template-made, with broken grammar ("26 Book", "How many did she
# graspeds?"), and its original repo states no licence


def missing(source: Source, raw: Path) -> list[str]:
    """The downloads of ``source`` not in ``raw`` yet."""
    return [d.path for d in source.downloads if not (raw / d.path).exists()]


def _checked(docs: Iterable[Doc], ctx: Context) -> Iterator[Doc]:
    """Every document once, long enough and with no travel topic left anywhere in it."""
    seen = set()
    for doc in docs:
        if doc.text in seen:
            ctx.drop("duplicate document")
        elif len(doc.text.split()) < MIN_DOC_WORDS:
            ctx.drop("short document")
        elif TRAVEL.search(doc.text):
            ctx.drop("travel topic")
        else:
            seen.add(doc.text)
            yield doc


def _capped(
    docs: list[Doc], cap: int | None, count: Callable[[str], int]
) -> tuple[list[Doc], int, int]:
    """The documents up to ``cap`` tokens, their tokens, and how many were cut."""
    kept, tokens = [], 0
    for doc in docs:
        if cap is not None and tokens >= cap:
            break
        kept.append(doc)
        tokens += count(doc.text)
    return kept, tokens, len(docs) - len(kept)


def build_source(
    name: str, raw: Path, out: Path, ctx: Context, count: Callable[[str], int]
) -> dict[str, Any]:
    """Convert one source into ``out/<name>.parquet`` (a ``text`` and a ``kind`` column) in a
    random order; return its report."""
    source = SOURCES[name]
    docs = shuffled(ctx.rng, list(_checked(source.convert(raw, ctx), ctx)))
    docs, tokens, cut = _capped(docs, source.cap, count)
    out.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": [d.text for d in docs], "kind": [d.kind for d in docs]})
    pq.write_table(table, out / f"{name}.parquet")
    report: dict[str, Any] = {
        "documents": len(docs),
        "tokens": tokens,
        "kinds": dict(Counter(d.kind for d in docs).most_common()),
        "licence": source.licence,
        "dropped": dict(ctx.dropped.most_common()),
    }
    if cut:
        report["cut_by_cap"] = cut
    return report


def build_tinytools(
    names: Iterable[str],
    raw: str | Path = defaults.RAW_DIR,
    out: str | Path = defaults.TINYTOOLS_DIR,
    tokenizer: str | Path | None = defaults.TOKENIZER_FILE,
    seed: int = 0,
    task_dir: str | Path = defaults.TASK_DIR,
    vocab_dir: str | Path = defaults.VOCAB_DIR,
) -> dict[str, Any]:
    """Build the named sources whose downloads are in ``raw``; return the report, which is also
    merged into ``out/report.json``."""
    raw, out = Path(raw), Path(out)
    evals = EvalSet.load(task_dir)
    sections = read_sections(Path(vocab_dir) / "nouns.txt")
    nouns = frozenset(w for words in sections.values() for w in words)
    person_nouns = frozenset(sections.get("people and family", []))
    first_names = read_words(Path(vocab_dir) / "names.txt")
    bpe = BPETokenizer.load(tokenizer) if tokenizer and Path(tokenizer).exists() else None

    def count(text: str) -> int:
        if bpe is None:
            return round(len(text.encode()) / BYTES_PER_TOKEN)
        return len(bpe.encode(text))

    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    sources = report.setdefault("sources", {})
    for name in names:
        if lacking := missing(SOURCES[name], raw):
            sources[name] = {"missing": lacking}
            continue
        rng = np.random.default_rng([seed, list(SOURCES).index(name)])
        ctx = Context(rng, evals, first_names, nouns, person_nouns)
        sources[name] = build_source(name, raw, out, ctx, count)
    report["eval_sets"] = list(evals.files)
    report["tokens_counted_with"] = str(tokenizer) if bpe else f"bytes / {BYTES_PER_TOKEN}"
    report["tokens"] = sum(s.get("tokens", 0) for s in sources.values())
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report
