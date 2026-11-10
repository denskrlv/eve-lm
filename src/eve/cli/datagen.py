"""``eve datagen``: generated fine-tuning data. ``run`` makes shards with local Ollama models and
Claude's review; ``report`` sums them up."""

import argparse
import json
from pathlib import Path

from eve import defaults
from eve.cli.common import Subcommands
from eve.progress import console, success, table, warn


def datagen_run(args: argparse.Namespace) -> None:
    """Make ``--shards`` new shards (and finish unfinished ones), then print the report."""
    from eve.data.datagen.llm import (
        LLMError,
        claude_generate,
        ollama_embed,
        ollama_generate,
        ollama_unload,
    )
    from eve.data.datagen.run import Models, load_context, run
    from eve.data.jsonl import read_lines

    topics_file = Path(defaults.DATAGEN_DIR) / "topics.txt"
    constraints_file = Path(defaults.DATAGEN_DIR) / "constraints.json"
    for path in (topics_file, constraints_file):
        if not path.exists():
            raise SystemExit(f"error: {path} is missing")
    embed = ollama_embed(args.embed_model)
    models = Models(
        ollama_generate(args.tools_model),
        ollama_generate(args.requests_model),
        claude_generate(args.claude_model, args.budget),
        embed,
        args.requests_model,
    )
    if args.tools_model != args.requests_model:
        models.between_stages = lambda: ollama_unload(args.tools_model)
    try:
        run(
            args.dir,
            args.shards,
            models,
            read_lines(topics_file),
            json.loads(constraints_file.read_text()),
            load_context(embed),
            args.parallel,
            args.seed,
            log=console.print,
        )
    except (LLMError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None
    _print_report(args.dir)


def _print_report(directory: str) -> None:
    from eve.data.datagen.run import report_rows

    rows = report_rows(directory)
    if not rows:
        warn(f"no shards in {directory}")
        return
    out = table(
        "shard",
        "status",
        "tools",
        "tools dropped",
        "rows",
        "written",
        "rejected",
        "fix rate",
        "dropped",
        "tok/s",
        "cost",
    )
    for r in rows:
        fix = "" if r["fix_rate"] is None else f"{r['fix_rate']:.1%}"
        cost = "" if r["cost"] is None else f"${r['cost']:.2f}"
        out.add_row(
            r["shard"],
            r["status"],
            str(r["tools"]),
            str(r["tools_dropped"]),
            str(r["rows"]),
            str(r["written"]),
            str(r["rejected"]),
            fix,
            "" if r["dropped"] is None else str(r["dropped"]),
            "" if r["tokens_per_s"] is None else f"{r['tokens_per_s']:.0f}",
            cost,
        )
    approved = [r for r in rows if r["status"] == "approved"]
    out.add_row(
        "total",
        f"{len(approved)} approved",
        str(sum(r["tools"] for r in rows)),
        str(sum(r["tools_dropped"] for r in rows)),
        str(sum(r["rows"] for r in approved)),
        str(sum(r["written"] for r in rows)),
        str(sum(r["rejected"] for r in rows)),
        "",
        "",
        "",
        f"${sum(r['cost'] or 0 for r in rows):.2f}",
    )
    console.print(out)
    success("shards", directory)


def datagen_report(args: argparse.Namespace) -> None:
    """Print one line per shard and the totals."""
    _print_report(args.dir)


def add_datagen(commands: Subcommands) -> None:
    """Register ``eve datagen run`` and ``eve datagen report``."""
    datagen = commands.add_parser("datagen", help="generate fine-tuning data with local models")
    sub = datagen.add_subparsers(dest="datagen_command", required=True)
    parser = sub.add_parser("run", help="make shards: tools, requests, checks, Claude's review")
    parser.set_defaults(run=datagen_run)
    parser.add_argument("--shards", type=int, default=1, help="new shards to make (0: only resume)")
    parser.add_argument("--dir", default=defaults.SHARDS_DIR)
    parser.add_argument("--tools-model", default=defaults.TOOLS_MODEL)
    parser.add_argument("--requests-model", default=defaults.REQUESTS_MODEL)
    parser.add_argument("--embed-model", default=defaults.EMBED_MODEL)
    parser.add_argument("--claude-model", default=None, help="Claude Code's default if not given")
    parser.add_argument("--budget", type=float, default=2.0, help="most US dollars per review")
    parser.add_argument("--parallel", type=int, default=2, help="requests in flight to Ollama")
    parser.add_argument("--seed", type=int, default=0)
    parser = sub.add_parser("report", help="one line per shard")
    parser.set_defaults(run=datagen_report)
    parser.add_argument("--dir", default=defaults.SHARDS_DIR)
