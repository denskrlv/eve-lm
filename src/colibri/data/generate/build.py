import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from colibri.config import ModelConfig
from colibri.data.generate.library import ToolSpec, load_specs
from colibri.data.generate.sampler import INTENT_MIX, ToolBank, sample
from colibri.data.generate.toolgen import generated_specs
from colibri.data.generate.validate import check_example, normalize
from colibri.data.jsonl import read_jsonl, read_lines, write_jsonl
from colibri.data.render import TaskExample, render
from colibri.data.schema import SchemaError, Tool
from colibri.tokenizer.bpe import BPETokenizer

CHAT_SPLITS = ("train", "val", "probe")
TEMPLATE_TRIES = 3  # samples drawn per template example wanted (repeats are skipped)
PROBE_HELDOUT_SHARE = 0.5  # probe scenarios whose correct tool is from a held-out domain


@dataclass
class Sources:
    train: list[ToolSpec]  # hand-written training domains + generated tools
    heldout: list[ToolSpec]  # never trained on
    chitchat: list[str]
    unsupported: list[str]

    @classmethod
    def load(cls, tools_dir="data/tools", generation_dir="data/generation", seed=0) -> "Sources":
        tools_dir, generation_dir = Path(tools_dir), Path(generation_dir)
        return cls(
            load_specs(tools_dir) + generated_specs(seed),
            load_specs(tools_dir / "heldout"),
            read_lines(generation_dir / "chitchat.txt"),
            read_lines(generation_dir / "unsupported.txt"),
        )

    @property
    def library(self) -> dict[str, Tool]:
        return {s.name: s.tool for s in self.train + self.heldout}

    def source_of(self, name: str) -> str:
        if any(s.name == name for s in self.heldout):
            return "heldout"
        if name not in self.library:
            return "older generated"  # a tool from an earlier generator, kept in a chat batch
        return "generated" if self.library[name].domain.startswith("generated") else "hand"


def template_examples(
    bank: ToolBank,
    n: int,
    rng: np.random.Generator,
    sources: Sources,
    pool: Sequence[int] | None = None,
) -> list[dict]:
    """``n`` template examples with valid labels; exact repeats are skipped."""
    rows, seen = [], set()
    for _ in range(n * TEMPLATE_TRIES):
        if len(rows) == n:
            break
        s = sample(bank, rng, pool=pool, chitchat=sources.chitchat, unsupported=sources.unsupported)
        ex = s.example
        key = (normalize(ex.utterance), tuple(sorted(t.name for t in ex.tools)), ex.intent)
        if key in seen or check_example(ex, s.implied):
            continue
        seen.add(key)
        rows.append({**ex.to_json(), "source": "template"})
    return rows


# ---- chat-written examples ------------------------------------------------------------------


def make_batch(
    sources: Sources, name: str, n: int, split: str, seed: int, ask: float | None = None
) -> dict:
    """Scenarios for a chat to write requests for (see data/generation/README.md).

    ``probe`` batches take the correct tool from the held-out domains half of the time.
    ``ask`` sets the share of ASK scenarios; the other intents keep their relative mix.
    """
    if split not in CHAT_SPLITS:
        raise ValueError(f"split must be one of {CHAT_SPLITS}")
    rng = np.random.default_rng(seed)
    specs = sources.train + (sources.heldout if split == "probe" else [])
    bank = ToolBank(specs)
    heldout = [bank.index[s.name] for s in sources.heldout] if split == "probe" else None
    mix = dict(INTENT_MIX)
    if ask is not None:
        rest = 1 - mix.pop("ask")
        mix = {k: v / rest * (1 - ask) for k, v in mix.items()} | {"ask": ask}
    scenarios = []
    for i in range(n):
        pool = heldout if heldout and rng.random() < PROBE_HELDOUT_SHARE else None
        intent = str(rng.choice(list(mix), p=list(mix.values())))
        s = sample(
            bank, rng, intent, pool, chitchat=sources.chitchat, unsupported=sources.unsupported
        )
        row = s.example.to_json(inline=True)  # whole schemas: the batch stays valid if tools change
        del row["utterance"]
        scenarios.append(
            {
                "id": f"{name}-{i:03d}",
                **row,
                "meaning": s.meaning,
                "missing": s.missing,
                "implied": sorted(s.implied),
            }
        )
    used = {t["name"]: Tool.from_dict(t) for s in scenarios for t in s["tools"]}
    schemas = {name: used[name].render() for name in sorted(used)}
    return {"batch": name, "split": split, "seed": seed, "schemas": schemas, "scenarios": scenarios}


def next_batch_name(batches_dir: str | Path) -> str:
    numbers = [int(p.stem.split("-")[1]) for p in Path(batches_dir).glob("batch-*.json")]
    return f"batch-{max(numbers, default=0) + 1:04d}"


def check_written(batch: dict, written: list[dict], library: dict[str, Tool]) -> tuple[list, list]:
    """Join written utterances to their scenarios. Returns (examples, rejects)."""
    scenarios = {s["id"]: s for s in batch["scenarios"]}
    examples, rejects = [], []
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
        if errors := check_example(ex, implied):
            rejects.append((line, errors))
        else:
            examples.append({**ex.to_json(inline=True), "source": "chat"})
    return examples, rejects


def chat_examples(sources: Sources, generation_dir="data/generation") -> tuple[dict, list]:
    """All written batches, by split, and everything that failed validation."""
    generation_dir = Path(generation_dir)
    by_split: dict[str, list[dict]] = {s: [] for s in CHAT_SPLITS}
    rejects = []
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


def _stats(rows: list[dict], sources: Sources) -> dict:
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


def build(
    out_dir: str | Path = "data/task",
    n_templates: int = 80_000,
    n_heldout: int = 2_000,
    seed: int = 0,
    tokenizer: BPETokenizer | None = None,
    cfg: ModelConfig | None = None,
    tools_dir="data/tools",
    generation_dir="data/generation",
    external_dir="data/external",
) -> dict:
    """Write the task splits as .jsonl files and return a report.

    - train: template examples + chat-written train batches + xLAM + MASSIVE train
    - val: chat-written val batches + MASSIVE dev. Only written by people or chats: template
      text is too easy for fitting calibration.
    - probe-candidates: chat-written probe batches, to be checked by hand into evals/probe-v1
    - heldout-templates: template examples whose correct tool is from a held-out domain
    - massive-test: MASSIVE's test partition, human-written
    External data is read from ``external_dir`` (see ``colibri data external``) if present.
    An utterance in a test or val split is removed from the splits below it.
    With a tokenizer, examples too long for the model are dropped.
    """
    sources = Sources.load(tools_dir, generation_dir)
    rng = np.random.default_rng(seed)
    train_bank = ToolBank(sources.train)
    all_bank = ToolBank(sources.train + sources.heldout)
    heldout_pool = [all_bank.index[s.name] for s in sources.heldout]

    chat, rejects = chat_examples(sources, generation_dir)

    def external(name: str) -> list[dict]:
        path = Path(external_dir) / f"{name}.jsonl"
        return read_jsonl(path) if path.exists() else []

    splits = {
        "train": template_examples(train_bank, n_templates, rng, sources)
        + chat["train"]
        + external("xlam")
        + external("massive-train"),
        "val": chat["val"] + external("massive-dev"),
        "probe-candidates": chat["probe"],
        "heldout-templates": template_examples(all_bank, n_heldout, rng, sources, heldout_pool),
        "massive-test": external("massive-test"),
    }

    removed = Counter()
    for keep, drop in [
        ("probe-candidates", "val"),
        ("probe-candidates", "train"),
        ("massive-test", "val"),
        ("massive-test", "train"),
        ("val", "train"),
    ]:
        seen = {normalize(r["utterance"]) for r in splits[keep]}
        before = len(splits[drop])
        splits[drop] = [r for r in splits[drop] if normalize(r["utterance"]) not in seen]
        removed[f"{drop} (in {keep})"] += before - len(splits[drop])

    too_long = Counter()
    if tokenizer is not None and cfg is not None:
        for name, rows in splits.items():
            fits = []
            for r in rows:
                try:
                    render(TaskExample.from_json(r, sources.library), tokenizer, cfg)
                    fits.append(r)
                except ValueError:
                    too_long[name] += 1
            splits[name] = fits

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
