import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eve import defaults
from eve.config import ModelConfig
from eve.data.generate.library import ToolSpec, load_specs
from eve.data.generate.sampler import INTENT_MIX, Sample, ToolBank, sample
from eve.data.generate.templates import sensible
from eve.data.generate.toolgen import generated_specs
from eve.data.generate.validate import check_example, normalize
from eve.data.jsonl import read_jsonl, read_lines, write_jsonl
from eve.data.render import TaskExample, render
from eve.data.schema import SchemaError, Tool
from eve.tokenizer.bpe import BPETokenizer

CHAT_SPLITS = ("train", "val", "probe")
Reject = tuple[dict[str, Any], list[str]]  # a written line and why it was rejected
TEMPLATE_TRIES = 3  # samples drawn per template example wanted (repeats are skipped)
PROBE_HELDOUT_SHARE = 0.5  # probe scenarios whose correct tool is from a held-out domain


@dataclass
class Sources:
    """What examples are drawn from: training and held-out tools, and request lists."""

    train: list[ToolSpec]  # hand-written training domains + generated tools
    heldout: list[ToolSpec]  # never trained on
    chitchat: list[str]
    unsupported: list[str]

    @classmethod
    def load(
        cls,
        tools_dir: str | Path = defaults.TOOLS_DIR,
        generation_dir: str | Path = defaults.GENERATION_DIR,
        seed: int = 0,
    ) -> "Sources":
        """The hand-written and generated tools, held-out tools, and line lists for NONE and
        UNSUPPORTED requests."""
        tools_dir, generation_dir = Path(tools_dir), Path(generation_dir)
        return cls(
            load_specs(tools_dir) + generated_specs(seed),
            load_specs(tools_dir / "heldout"),
            read_lines(generation_dir / "chitchat.txt"),
            read_lines(generation_dir / "unsupported.txt"),
        )

    @property
    def library(self) -> dict[str, Tool]:
        """Every known tool by name, training and held-out."""
        return {s.name: s.tool for s in self.train + self.heldout}

    def source_of(self, name: str) -> str:
        """Where a tool comes from: "hand", "generated", "heldout" or "older generated"."""
        if any(s.name == name for s in self.heldout):
            return "heldout"
        if name not in self.library:
            return "older generated"  # a tool from an earlier generator, kept in a chat batch
        return "generated" if self.library[name].domain.startswith("generated") else "hand"


def template_examples(
    bank: ToolBank,
    count: int,
    rng: np.random.Generator,
    sources: Sources,
    pool: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """``count`` template examples with valid labels; exact repeats are skipped."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for _ in range(count * TEMPLATE_TRIES):
        if len(rows) == count:
            break
        drawn = sample(
            bank, rng, pool=pool, chitchat=sources.chitchat, unsupported=sources.unsupported
        )
        ex = drawn.example
        key = (normalize(ex.utterance), tuple(sorted(t.name for t in ex.tools)), ex.intent)
        if key in seen or check_example(ex, drawn.implied):
            continue
        seen.add(key)
        rows.append({**ex.to_json(), "source": "template"})
    return rows


# ---- chat-written examples ------------------------------------------------------------------


def _intent_mix(ask: float | None) -> dict[str, float]:
    """``INTENT_MIX``, or with ASK at ``ask`` and the other intents scaled to share the rest."""
    mix = dict(INTENT_MIX)
    if ask is not None:
        rest = 1 - mix.pop("ask")
        mix = {k: v / rest * (1 - ask) for k, v in mix.items()} | {"ask": ask}
    return mix


def _scenario(drawn: Sample, scenario_id: str) -> dict[str, Any]:
    """A drawn example as a scenario for chat writers: everything but the utterance."""
    row = drawn.example.to_json(inline=True)  # whole schemas: valid even if tools change
    del row["utterance"]
    return {
        "id": scenario_id,
        **row,
        "meaning": drawn.meaning,
        "missing": drawn.missing,
        "implied": sorted(drawn.implied),
    }


def make_batch(
    sources: Sources, name: str, count: int, split: str, seed: int, ask: float | None = None
) -> dict[str, Any]:
    """Scenarios for a chat to write requests for (see data/generation/README.md).

    ``probe`` batches take the correct tool from the held-out domains half of the time.
    ``ask`` sets the share of ASK scenarios; the other intents keep their relative mix.
    """
    if split not in CHAT_SPLITS:
        raise ValueError(f"split must be one of {CHAT_SPLITS}")
    rng = np.random.default_rng(seed)
    specs = sources.train + (sources.heldout if split == "probe" else [])
    bank = ToolBank(specs)
    heldout = [bank.index[spec.name] for spec in sources.heldout] if split == "probe" else None
    mix = _intent_mix(ask)
    scenarios = []
    for i in range(count):
        pool = heldout if heldout and rng.random() < PROBE_HELDOUT_SHARE else None
        intent = str(rng.choice(list(mix), p=list(mix.values())))
        drawn = sample(
            bank, rng, intent, pool, chitchat=sources.chitchat, unsupported=sources.unsupported
        )
        scenarios.append(_scenario(drawn, f"{name}-{i:03d}"))
    used = {t["name"]: Tool.from_dict(t) for scenario in scenarios for t in scenario["tools"]}
    schemas = {name: used[name].render() for name in sorted(used)}
    return {"batch": name, "split": split, "seed": seed, "schemas": schemas, "scenarios": scenarios}


def next_batch_name(batches_dir: str | Path) -> str:
    """The next unused ``batch-NNNN`` name in ``batches_dir``."""
    numbers = [int(p.stem.split("-")[1]) for p in Path(batches_dir).glob("batch-*.json")]
    return f"batch-{max(numbers, default=0) + 1:04d}"


def _scenario_errors(utterance: str, scenario: dict[str, Any]) -> list[str]:
    """Problems ``check_example`` can't see: a copied ``meaning``, and calls that ``sensible``
    refuses (scenarios sampled before it existed)."""
    errors = []
    meaning = scenario.get("meaning")
    if meaning and normalize(utterance) == normalize(meaning):
        errors.append("the utterance copies the meaning; rephrase it")
    for call in scenario.get("calls", ()):
        if not sensible(call.get("args", {})):
            errors.append(f"call to {call['tool']} is a conversion no one asks for; re-sample it")
    return errors


def check_written(
    batch: dict[str, Any], written: list[dict[str, Any]], library: dict[str, Tool]
) -> tuple[list[dict[str, Any]], list[Reject]]:
    """Join written utterances to their scenarios. Returns (examples, rejects)."""
    scenarios = {s["id"]: s for s in batch["scenarios"]}
    examples: list[dict[str, Any]] = []
    rejects: list[Reject] = []
    for line in written:
        scenario = scenarios.get(line.get("id"))
        if scenario is None:
            rejects.append((line, ["unknown scenario id"]))
            continue
        try:
            ex = TaskExample.from_json(
                {**scenario, "utterance": line.get("utterance", "")}, library
            )
        except (SchemaError, KeyError) as e:
            rejects.append((line, [str(e)]))
            continue
        implied = {tuple(x) for x in scenario.get("implied", ())}
        if errors := check_example(ex, implied) + _scenario_errors(ex.utterance, scenario):
            rejects.append((line, errors))
        else:
            examples.append({**ex.to_json(inline=True), "source": "chat"})
    return examples, rejects


def chat_examples(
    sources: Sources, generation_dir: str | Path = defaults.GENERATION_DIR
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, dict[str, Any], list[str]]]]:
    """All written batches, by split, and everything that failed validation."""
    generation_dir = Path(generation_dir)
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in CHAT_SPLITS}
    rejects: list[tuple[str, dict[str, Any], list[str]]] = []
    for path in sorted((generation_dir / "batches").glob("batch-*.json")):
        written = generation_dir / "written" / f"{path.stem}.jsonl"
        if not written.exists():
            continue
        batch = json.loads(path.read_text())
        examples, bad = check_written(batch, read_jsonl(written), sources.library)
        by_split[batch["split"]] += examples
        rejects += [(path.stem, line, errors) for line, errors in bad]
    return by_split, rejects


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
                sources.source_of(c["tool"])
                if r.get("source") in ("template", "chat")
                else r["source"]
                for r in rows
                for c in r["calls"]
            )
        ),
    }


Splits = dict[str, list[dict[str, Any]]]
# (keep, drop): an utterance in ``keep`` is removed from ``drop``, so test and val stay unseen.
DEDUP_ORDER = (
    ("probe-candidates", "val"),
    ("probe-candidates", "train"),
    ("massive-test", "val"),
    ("massive-test", "train"),
    ("val", "train"),
)


def _external(external_dir: Path, name: str) -> list[dict[str, Any]]:
    """A converted external dataset (``eve data external``), or nothing if absent."""
    path = external_dir / f"{name}.jsonl"
    return read_jsonl(path) if path.exists() else []


def _assemble(
    sources: Sources,
    chat: Splits,
    rng: np.random.Generator,
    n_templates: int,
    n_heldout: int,
    external_dir: Path,
) -> Splits:
    """Every split from its sources (see ``build``)."""
    train_bank = ToolBank(sources.train)
    all_bank = ToolBank(sources.train + sources.heldout)
    heldout_pool = [all_bank.index[s.name] for s in sources.heldout]
    return {
        "train": template_examples(train_bank, n_templates, rng, sources)
        + chat["train"]
        + _external(external_dir, "xlam")
        + _external(external_dir, "massive-train"),
        "val": chat["val"] + _external(external_dir, "massive-dev"),
        "probe-candidates": chat["probe"],
        "heldout-templates": template_examples(all_bank, n_heldout, rng, sources, heldout_pool),
        "massive-test": _external(external_dir, "massive-test"),
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
    n_templates: int = defaults.TEMPLATE_EXAMPLES,
    n_heldout: int = defaults.HELDOUT_EXAMPLES,
    seed: int = 0,
    tokenizer: BPETokenizer | None = None,
    cfg: ModelConfig | None = None,
    tools_dir: str | Path = defaults.TOOLS_DIR,
    generation_dir: str | Path = defaults.GENERATION_DIR,
    external_dir: str | Path = defaults.EXTERNAL_DIR,
) -> dict[str, Any]:
    """Write the task splits as .jsonl files and return a report.

    - train: template examples + chat-written train batches + xLAM + MASSIVE train
    - val: chat-written val batches + MASSIVE dev. Only written by people or chats: template
      text is too easy for fitting calibration.
    - probe-candidates: chat-written probe batches, to be checked by hand into evals/probe-v1
    - heldout-templates: template examples whose correct tool is from a held-out domain
    - massive-test: MASSIVE's test partition, human-written
    External data is read from ``external_dir`` (see ``eve data external``) if present.
    An utterance in a test or val split is removed from the splits below it.
    With a tokenizer, examples too long for the model are dropped.
    """
    sources = Sources.load(tools_dir, generation_dir)
    rng = np.random.default_rng(seed)
    chat, rejects = chat_examples(sources, generation_dir)
    splits = _assemble(sources, chat, rng, n_templates, n_heldout, Path(external_dir))
    removed = _remove_duplicates(splits)
    too_long: Counter[str] = Counter()
    if tokenizer is not None and cfg is not None:
        too_long = _drop_too_long(splits, sources.library, tokenizer, cfg)

    out_dir = Path(out_dir)
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    report = {
        "splits": {name: _stats(rows, sources) for name, rows in splits.items()},
        "tools": {"train": len(sources.train), "heldout": len(sources.heldout)},
        "rejected_chat_lines": len(rejects),
        "removed_duplicates": dict(removed),
        "too_long": dict(too_long),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
