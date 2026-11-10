"""``eve datagen run``: make shards from topics to approved requests, resuming any left unfinished.

The stages go one at a time over all the shards a run holds: stage A (tools) for all, then
stage B (requests) for all, so only one local model needs to be in memory. A shard whose requests
are done goes to Claude's review at once, in the background, while stage B goes on.
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eve import defaults
from eve.data.datagen.llm import Embed, Generate
from eve.data.datagen.meaning import MeaningIndex, blocked_index, library_index
from eve.data.datagen.requests import RowStats, draw_cards, write_requests
from eve.data.datagen.review import review_shard, shard_report
from eve.data.datagen.shard import (
    DONE,
    TOOLS_PER_SHARD,
    TOPICS_PER_SHARD,
    Shard,
    claim,
    new_shard,
    release,
    shards,
)
from eve.data.datagen.tools import Judge, ToolStats, invent_tools
from eve.data.generate.build import Sources
from eve.data.generate.novel import load_novel
from eve.data.generate.validate import normalize
from eve.data.schema import Tool, load_library


@dataclass
class Models:
    """The models of a run."""

    tools: Generate  # stage A
    requests: Generate  # stage B
    review: Generate  # Claude
    embed: Embed
    requests_name: str  # recorded as each row's writer
    between_stages: Callable[[], None] = lambda: None  # e.g. unload the stage A model


@dataclass
class Context:
    """What the checks compare new tools against, loaded once per run."""

    blocked: MeaningIndex  # novel-tools and held-out tools
    blocked_tools: list[Tool]
    library: MeaningIndex
    capabilities: dict[str, str]  # training tool name -> capability
    reserved_names: set[str]  # names no generated tool may take


def load_context(
    embed: Embed,
    tools_dir: str | Path = defaults.TOOLS_DIR,
    novel_dir: str | Path = defaults.NOVEL_TOOLS_DIR,
) -> Context:
    """The blocked tools and the training library, embedded."""
    novel = list(load_novel(novel_dir)[0].values())
    heldout = list(load_library(Path(tools_dir) / "heldout").values())
    specs = Sources.load(tools_dir).train
    names = {s.name for s in specs} | {t.name for t in novel + heldout}
    return Context(
        blocked_index(embed, novel_dir, Path(tools_dir) / "heldout"),
        novel + heldout,
        library_index(embed, specs),
        {s.name: s.capability for s in specs},
        names,
    )


def _stage_a(
    shard: Shard, root: Path, models: Models, ctx: Context, seeds: Sequence[str], seed: int
) -> None:
    others = [s for s in shards(root) if s.number != shard.number]
    accepted = MeaningIndex(models.embed)
    earlier = [t for s in others for t in s.tools()]
    accepted.add_many([(t.names[0], list(t.descriptions)) for t in earlier])
    taken = ctx.reserved_names | {n for t in earlier for n in t.names}
    judge = Judge(ctx.blocked, accepted, ctx.library, ctx.capabilities, taken)
    rng = np.random.default_rng([seed, shard.number])
    per_topic = TOOLS_PER_SHARD // TOPICS_PER_SHARD
    tools, stats = invent_tools(models.tools, shard.topics(), judge, rng, seeds, per_topic)
    shard.save_tools(tools)
    shard.write_json("stats.json", {"tools": stats.to_dict()})
    shard.set_status("tools")


def _stage_b(
    shard: Shard,
    root: Path,
    models: Models,
    constraints: Mapping[str, Sequence[str]],
    parallel: int,
    seed: int,
) -> None:
    seen = {
        normalize(r.utterance) for s in shards(root) if s.number != shard.number for r in s.rows()
    }
    tools = shard.tools()
    rng = np.random.default_rng([seed, shard.number, 1])
    cards = [draw_cards(t, constraints, rng) for t in tools]
    with ThreadPoolExecutor(max(1, parallel)) as pool:
        done = list(
            pool.map(
                lambda tc: write_requests(
                    models.requests, tc[0], tc[1], seen, models.requests_name
                ),
                zip(tools, cards, strict=True),
            )
        )
    stats = RowStats()
    for _, tool_stats in done:
        stats.add(tool_stats)
    shard.save_rows([row for rows, _ in done for row in rows])
    shard.write_json(
        "stats.json", {**shard.read_json("stats.json", {}), "requests": stats.to_dict()}
    )
    shard.set_status("requests")


def _review(shard: Shard, models: Models, ctx: Context) -> None:
    tools, rows = shard.tools(), shard.rows()
    result = review_shard(models.review, tools, rows, ctx.blocked_tools)
    names = {t.id: t.names[0] for t in tools}
    shard.write_json(
        "review.json",
        {"before": [r.to_dict(names.get(r.tool, "")) for r in rows], "reasons": result.reasons},
    )
    shard.save_tools(result.tools)
    shard.save_rows(result.rows)
    stats = shard.read_json("stats.json", {})
    report = shard_report(
        ToolStats.from_dict(stats.get("tools", {})),
        RowStats.from_dict(stats.get("requests", {})),
        result,
    )
    report["approved"] = report["approved"] and bool(result.rows)
    shard.write_json("report.json", report)
    shard.set_status("approved" if report["approved"] else "flagged")


def run(
    root: str | Path,
    count: int,
    models: Models,
    topics: Sequence[str],
    constraints: Mapping[str, Sequence[str]],
    context: Context,
    parallel: int = 2,
    seed: int = 0,
    log: Callable[[str], None] = lambda line: None,
) -> list[Shard]:
    """Finish every unfinished shard under ``root`` no other process holds, then make ``count``
    new ones. Returns the shards this run worked on. A model failure (``LLMError``) stops the run;
    every shard keeps its last finished stage and the next run picks it up."""
    root = Path(root)
    held = [s for s in shards(root) if s.status() not in DONE and claim(s)]
    try:
        for _ in range(count):
            shard = new_shard(root, topics)
            if claim(shard):
                held.append(shard)
        for shard in held:
            if shard.status() == "new":
                _stage_a(shard, root, models, context, constraints["seeds"], seed)
                log(f"{shard.path.name}: {len(shard.tools())} tools")
        if any(s.status() == "tools" for s in held):
            models.between_stages()
        with ThreadPoolExecutor(1) as reviews:
            pending: list[Future[None]] = []
            for shard in held:
                if shard.status() == "tools":
                    _stage_b(shard, root, models, constraints, parallel, seed)
                    log(f"{shard.path.name}: {len(shard.rows())} requests")
                if shard.status() == "requests":
                    pending.append(reviews.submit(_review, shard, models, context))
            for future in pending:
                future.result()
        for shard in held:
            log(f"{shard.path.name}: {shard.status()}")
    finally:
        for shard in held:
            release(shard)
    return held


def report_rows(root: str | Path) -> list[dict[str, Any]]:
    """One summary per shard under ``root``, for ``eve datagen report``."""
    out = []
    for shard in shards(Path(root)):
        report = shard.read_json("report.json", {})
        stats = shard.read_json("stats.json", {})
        rows = report.get("rows") or stats.get("requests", {})
        review = report.get("review", {})
        out.append(
            {
                "shard": shard.path.name,
                "status": shard.status(),
                "tools": len(shard.tools()),
                "rows": len(shard.rows()),
                "written": rows.get("written", 0),
                "rejected": sum(rows.get("rejected", {}).values()),
                "fix_rate": review.get("fix_rate"),
                "dropped": review.get("dropped"),
                "tokens_per_s": report.get("speed", {}).get("requests_tokens_per_s"),
                "cost": review.get("cost"),
            }
        )
    return out
