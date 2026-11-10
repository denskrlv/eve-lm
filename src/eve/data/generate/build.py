import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eve import defaults
from eve.config import ModelConfig
from eve.data.generate.library import ToolSpec, load_specs
from eve.data.generate.novel import novel_examples
from eve.data.generate.sampler import ToolBank, sample
from eve.data.generate.toolgen import generated_specs
from eve.data.generate.validate import check_example, normalize
from eve.data.jsonl import read_jsonl, read_lines, write_jsonl
from eve.data.render import TaskExample, render
from eve.data.schema import Tool
from eve.tokenizer.bpe import BPETokenizer

TEMPLATE_TRIES = 3  # samples drawn per template example wanted (repeats are skipped)


@dataclass
class Sources:
    """What examples are drawn from: training and held-out tools, and request lists."""

    train: list[ToolSpec]  # hand-written training domains + generated tools
    heldout: list[ToolSpec]  # never trained on
    chitchat: list[str]
    unsupported: list[str]
    heldout_chitchat: list[str]  # only for the held-out set, so its NONE recall isn't memorized
    heldout_unsupported: list[str]

    @classmethod
    def load(
        cls,
        tools_dir: str | Path = defaults.TOOLS_DIR,
        generation_dir: str | Path = defaults.GENERATION_DIR,
        seed: int = 0,
    ) -> "Sources":
        """The hand-written and generated tools, held-out tools, and line lists for NONE and
        UNSUPPORTED requests: the training ones, and the held-out ones in ``heldout/``."""
        tools_dir, generation_dir = Path(tools_dir), Path(generation_dir)
        return cls(
            load_specs(tools_dir) + generated_specs(seed),
            load_specs(tools_dir / "heldout"),
            read_lines(generation_dir / "chitchat.txt"),
            read_lines(generation_dir / "unsupported.txt"),
            read_lines(generation_dir / "heldout" / "chitchat.txt"),
            read_lines(generation_dir / "heldout" / "unsupported.txt"),
        )

    @property
    def library(self) -> dict[str, Tool]:
        """Every known tool by name, training and held-out."""
        return {s.name: s.tool for s in self.train + self.heldout}

    def source_of(self, name: str) -> str:
        """Where a tool comes from: "hand", "generated" or "heldout"."""
        if any(s.name == name for s in self.heldout):
            return "heldout"
        return "generated" if self.library[name].domain.startswith("generated") else "hand"


def template_examples(
    bank: ToolBank,
    count: int,
    rng: np.random.Generator,
    sources: Sources,
    pool: Sequence[int] | None = None,
    heldout: bool = False,
) -> list[dict[str, Any]]:
    """``count`` template examples with valid labels; exact repeats are skipped. ``heldout``
    takes NONE and UNSUPPORTED requests from the held-out lines instead of the training ones."""
    chitchat = sources.heldout_chitchat if heldout else sources.chitchat
    unsupported = sources.heldout_unsupported if heldout else sources.unsupported
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for _ in range(count * TEMPLATE_TRIES):
        if len(rows) == count:
            break
        drawn = sample(bank, rng, pool=pool, chitchat=chitchat, unsupported=unsupported)
        ex = drawn.example
        # small talk comes from a short list: new tools alone don't make it a new example
        tools = () if ex.intent == "none" else tuple(sorted(t.name for t in ex.tools))
        key = (normalize(ex.utterance), tools, ex.intent)
        if key in seen or check_example(ex, drawn.implied):
            continue
        seen.add(key)
        rows.append({**ex.to_json(), "source": "template"})
    return rows


# ---- the whole dataset ----------------------------------------------------------------------


def _stats(rows: list[dict[str, Any]], sources: Sources) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "intent": dict(Counter(r["intent"] for r in rows)),
        "calls": dict(sorted(Counter(len(r["calls"]) for r in rows).items())),
        "tools_in_prompt": dict(sorted(Counter(len(r["tools"]) for r in rows).items())),
        "source": dict(Counter(r.get("source", "template") for r in rows)),
        "called_tool_source": dict(
            Counter(
                sources.source_of(c["tool"]) if r.get("source") == "template" else r["source"]
                for r in rows
                for c in r["calls"]
            )
        ),
    }


Splits = dict[str, list[dict[str, Any]]]
TRAIN_SPLITS = ("train",)  # written to the task directory; every other split to the evals one
# (keep, drop): an utterance in ``keep`` is removed from ``drop``, so test and val stay unseen.
DEDUP_ORDER = (
    ("novel-tools", "val"),
    ("novel-tools", "train"),
    ("massive-test", "val"),
    ("massive-test", "train"),
    ("val", "train"),
    ("train", "heldout-templates"),  # its own lines can still match external train rows
)


def _external(external_dir: Path, name: str) -> list[dict[str, Any]]:
    """A converted external dataset (``eve data external``), or nothing if absent."""
    path = external_dir / f"{name}.jsonl"
    return read_jsonl(path) if path.exists() else []


def _assemble(
    sources: Sources,
    rng: np.random.Generator,
    n_templates: int,
    n_heldout: int,
    external_dir: Path,
    novel_dir: Path,
) -> Splits:
    """Every split from its sources (see ``build``)."""
    train_bank = ToolBank(sources.train)
    all_bank = ToolBank(sources.train + sources.heldout)
    heldout_pool = [all_bank.index[s.name] for s in sources.heldout]
    return {
        "train": template_examples(train_bank, n_templates, rng, sources)
        + _external(external_dir, "xlam")
        + _external(external_dir, "massive-train"),
        "val": _external(external_dir, "massive-dev"),
        "heldout-templates": template_examples(
            all_bank, n_heldout, rng, sources, heldout_pool, heldout=True
        ),
        "massive-test": _external(external_dir, "massive-test"),
        "novel-tools": novel_examples(rng, sources.library, novel_dir),
    }


def _remove_duplicates(splits: Splits) -> Counter[str]:
    """Remove utterances that also appear in an earlier split (``DEDUP_ORDER``), in place.
    Returns how many were removed from where."""
    removed: Counter[str] = Counter()
    for keep, drop in DEDUP_ORDER:
        seen = {normalize(r["utterance"]) for r in splits[keep]}
        before = len(splits[drop])
        splits[drop] = [r for r in splits[drop] if normalize(r["utterance"]) not in seen]
        removed[f"{drop} (in {keep})"] += before - len(splits[drop])
    return removed


def _add_datagen(splits: Splits, sources: Sources, root: Path, rng: np.random.Generator) -> int:
    """Append the approved generated rows to train, except those whose request is already in a
    split. Returns how many were left out."""
    from eve.data.datagen.examples import datagen_examples, load_approved

    if not root.is_dir():
        raise FileNotFoundError(f"--datagen: no shards directory at {root}")
    tools, rows = load_approved(root)
    if not rows:
        logging.getLogger(__name__).warning("--datagen: no approved shards in %s", root)
    generated = datagen_examples(tools, rows, sources.train, rng)
    taken = {normalize(r["utterance"]) for split in splits.values() for r in split}
    kept = [r for r in generated if normalize(r["utterance"]) not in taken]
    splits["train"] += kept
    return len(generated) - len(kept)


def _drop_too_long(
    splits: Splits, library: dict[str, Tool], tokenizer: BPETokenizer, cfg: ModelConfig
) -> Counter[str]:
    """Remove examples too long for the model's encoder or decoder, in place. Returns how many
    were removed per split."""
    too_long: Counter[str] = Counter()
    for name, rows in splits.items():
        fits = []
        for row in rows:
            try:
                render(TaskExample.from_json(row, library), tokenizer, cfg)
                fits.append(row)
            except ValueError:
                too_long[name] += 1
        splits[name] = fits
    return too_long


def build(
    out_dir: str | Path = defaults.TASK_DIR,
    evals_dir: str | Path = defaults.EVALS_DIR,
    n_templates: int = defaults.TEMPLATE_EXAMPLES,
    n_heldout: int = defaults.HELDOUT_EXAMPLES,
    seed: int = 0,
    tokenizer: BPETokenizer | None = None,
    cfg: ModelConfig | None = None,
    tools_dir: str | Path = defaults.TOOLS_DIR,
    generation_dir: str | Path = defaults.GENERATION_DIR,
    external_dir: str | Path = defaults.EXTERNAL_DIR,
    novel_dir: str | Path = defaults.NOVEL_TOOLS_DIR,
    datagen_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the splits as .jsonl files, train to ``out_dir`` and the sets models are scored on
    to ``evals_dir``, and return a report (also ``out_dir/report.json``).

    - train: template examples + xLAM + MASSIVE train
    - val: MASSIVE dev. Only written by people: template text is too easy for fitting
      calibration.
    - heldout-templates: template examples whose correct tool is from a held-out domain; it
      also fits the new-tool temperatures (``eve calibrate``)
    - massive-test: MASSIVE's test partition, human-written
    - novel-tools: made-up tools with hand-written requests, from ``novel_dir`` (see
      ``eve.data.generate.novel``)
    External data is read from ``external_dir`` (see ``eve data external``) if present.
    An utterance in a test or val split is removed from the splits below it.
    heldout-templates takes its NONE and UNSUPPORTED requests from ``generation_dir/heldout``,
    lines train never has.
    With ``datagen_dir``, the approved generated shards there are added to train (see
    ``eve.data.datagen.examples``), after the duplicates are removed: a generated request that is
    in any other split is dropped instead, so the evals don't change.
    With a tokenizer, examples too long for the model are dropped.
    """
    sources = Sources.load(tools_dir, generation_dir)
    rng = np.random.default_rng(seed)
    splits = _assemble(sources, rng, n_templates, n_heldout, Path(external_dir), Path(novel_dir))
    removed = _remove_duplicates(splits)
    if datagen_dir is not None:
        removed["train (datagen, already in a split)"] = _add_datagen(
            splits, sources, Path(datagen_dir), np.random.default_rng([seed, 1])
        )
    too_long: Counter[str] = Counter()
    if tokenizer is not None and cfg is not None:
        too_long = _drop_too_long(splits, sources.library, tokenizer, cfg)

    out_dir, evals_dir = Path(out_dir), Path(evals_dir)
    for name, rows in splits.items():
        write_jsonl((out_dir if name in TRAIN_SPLITS else evals_dir) / f"{name}.jsonl", rows)
    report = {
        "splits": {name: _stats(rows, sources) for name, rows in splits.items()},
        "tools": {"train": len(sources.train), "heldout": len(sources.heldout)},
        "removed_duplicates": dict(removed),
        "too_long": dict(too_long),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
