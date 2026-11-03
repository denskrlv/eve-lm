"""Commands that query a trained model: ``ask``, ``evaluate``, ``calibrate`` and ``fill``."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rich.text import Text

from eve import defaults
from eve.cli.common import TRAIN_DEFAULTS, Subcommands, add_device_arg, add_json_arg, add_model_arg
from eve.config import INTENTS
from eve.progress import console, panel, rows, success, table
from eve.runtime import pick_device

if TYPE_CHECKING:
    from rich.panel import Panel
    from rich.table import Table

    from eve.api import Call, Response
    from eve.calibrate import Reliability, TaskSet
    from eve.data.schema import Tool
    from eve.evaluate import SetResult
    from eve.infer.predict import Calibrations
    from eve.model.checkpoint import Checkpoint
    from eve.model.eve import Eve


INTENTS_SHOWN, TOOLS_SHOWN = 4, 3  # answers eve ask shows for the intent and per call


SURE, UNSURE = 0.8, 0.5  # eve ask colors a top answer green from SURE, yellow from UNSURE


SCORE_COLUMNS = (  # eve evaluate's table: (label, key in the scores)
    ("macro", "macro_recall"),
    *((intent[:5], f"recall_{intent}") for intent in INTENTS),  # "unsupported" -> "unsup"
    ("tools", "tools_acc"),
    ("exact", "exact_acc"),
    ("values", "value_acc"),
    ("full", "full_acc"),
)


def _print_fill_report(
    model_path: str, meta: dict[str, Any], device: torch.device, filled: list[str]
) -> None:
    """Which checkpoint filled the holes and how far it was trained, then each filled text."""
    from eve.tasks.span import HIGHLIGHTS

    trained = f"{meta['step']:,} steps" if "step" in meta else "?"
    if "val_lm" in meta:
        trained += f" · val loss {meta['val_lm']:.3f}"
    about = [("checkpoint", model_path), ("trained", trained), ("device", str(device))]
    console.print(panel("eve fill", about))
    texts = [Text(line) for line in filled]
    for text in texts:
        for pattern, style in HIGHLIGHTS:
            text.highlight_regex(pattern, style)
    console.print(rows([(str(i), text) for i, text in enumerate(texts, 1)], indent=2))
    console.print()


def fill(args: argparse.Namespace) -> None:
    """Ask a pretrained model to fill the <x0>, <x1>… holes in a sentence."""
    from eve import progress
    from eve.model.checkpoint import load_eve
    from eve.tasks.span import SAMPLES, fill_in

    texts = args.text or list(SAMPLES)
    if bad := [t for t in texts if "<x0>" not in t]:
        raise SystemExit(f"put a hole like <x0> in the text: {bad[0]!r}")
    device = pick_device(args.device)
    with progress.bar(len(texts), "fill", "prompts", leave=False) as bar:
        bar.set_stats(f"loading {Path(args.model).name}")
        ckpt = load_eve(args.model, device)
        filled = []
        for text in texts:
            bar.set_stats("decoding")
            filled.append(fill_in(ckpt.model, ckpt.tokenizer, text, args.max_new))
            bar.update()

    _print_fill_report(str(args.model), ckpt.meta, device, filled)


def _tools(args: argparse.Namespace) -> list["Tool"]:
    """The tools for ``eve ask``: names from the library and/or schemas from JSON files."""
    from eve.api import EveError
    from eve.api.tools import to_tool
    from eve.data.generate.build import Sources

    tools = []
    if args.tools:
        library = Sources.load().library
        names = [name.strip() for name in args.tools.split(",") if name.strip()]
        if unknown := [name for name in names if name not in library]:
            raise SystemExit(f"unknown tools {unknown}; see data/tools/*.json or use --schema")
        tools += [library[name] for name in names]
    for path in args.schema or []:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"{path}: {error}") from None
        if isinstance(data, dict) and "tools" not in data:
            raise SystemExit(f'{path}: no "tools" list')
        schemas = data["tools"] if isinstance(data, dict) else data  # a data/tools file, or a list
        try:
            tools += [to_tool(t)[0] for t in schemas]  # Eve, OpenAI or MCP schemas
        except EveError as error:
            raise SystemExit(f"{path}: {error}") from None
    if not tools:
        raise SystemExit("give the tools with --tools a,b,c and/or --schema FILE")
    return tools


def _confidence(prob: float) -> str:
    """The color of an answer this likely: green when sure, yellow when unsure, red below."""
    return "green" if prob >= SURE else "yellow" if prob >= UNSURE else "red"


def _ranked(probs: dict[str, float], count: int) -> Text:
    """The ``count`` most likely answers with their probabilities: the first in bold, colored
    by how likely it is, the others dimmed."""
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:count]
    return Text(" · ", style="dim").join(
        Text(f"{k} {p:.0%}", style=f"bold {_confidence(p)}" if i == 0 else "dim")
        for i, (k, p) in enumerate(top)
    )


def _values(call: "Call") -> Text:
    """``room kitchen 98% · on false 99% · brightness –``: each argument and what the decoder
    wrote, colored by how likely it is; ``?`` for one to ask the user for, ``✗`` if invalid."""
    parts = []
    for name, prob in call.arg_confidence.items():
        part = Text(f"{name} ")
        if name in call.args:
            value = call.args[name]
            shown = value if isinstance(value, str) else json.dumps(value)
            part.append(str(shown), style=f"bold {_confidence(prob)}")
        elif name in call.missing:
            part.append("?", style="bold yellow")
        elif name in call.invalid:
            part.append("✗", style="bold red")
        else:
            parts.append(Text(f"{name} –", style="dim"))
            continue
        part.append(f" {prob:.0%}", style="dim")
        parts.append(part)
    return Text(" · ", style="dim").join(parts) if parts else Text("no arguments", style="dim")


def _print_response(response: "Response", as_json: bool) -> None:
    """One response: a JSON line, or the intent and each call with its likeliest tools and the
    arguments written for it."""
    if as_json:
        print(json.dumps(response.to_dict()))
        return
    items = [("intent", _ranked(response.intents, INTENTS_SHOWN))]
    for i, call in enumerate(response.calls):
        items.append((f"call {i + 1}", _ranked(response.slots[i], TOOLS_SHOWN)))
        items.append(("", _values(call)))
    console.print(rows(items, indent=2))
    console.print()


def _ask_interactively(show: Callable[[str], None]) -> None:
    """Read requests from the terminal and show the answers, until an empty line or Ctrl-D."""
    while True:
        try:
            text = console.input(Text("› ", style="bold cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            break
        show(text)


def ask(args: argparse.Namespace) -> None:
    """What a fine-tuned model decides for a typed request, through the public
    API (``eve.Eve``): the intent, each call with its argument values, and the probabilities,
    calibrated if ``eve calibrate --save`` wrote temperatures (familiar or new tools, chosen
    per request)."""
    from eve.api import Eve, EveError

    tools = _tools(args)
    try:
        eve = Eve.load(args.model, device=args.device or "auto")
        eve.add_tools(tools)  # a schema Eve can't read fails before the first request
    except EveError as error:
        raise SystemExit(str(error)) from None

    def show(text: str) -> None:
        try:
            _print_response(eve.respond(text), args.json)
        except EveError as error:
            console.print(Text(str(error), style="red"))

    if args.text:
        for text in args.text:
            if not args.json:
                console.print(Text(text, style="bold"))
            show(text)
        return
    names = ", ".join(t.name for t in tools)
    calibrated = eve.info["calibrated"]
    probs = "calibrated (familiar or new tools)" if calibrated else "as trained: see eve calibrate"
    items = [("model", str(args.model)), ("tools", names), ("probs", probs)]
    console.print(panel("eve ask", items))
    console.print(Text("type a request · empty line or Ctrl-D to quit\n", style="dim"))
    _ask_interactively(show)


def _evaluate_panel(
    args: argparse.Namespace, meta: dict[str, Any], device: torch.device
) -> "Panel":
    """Which model is scored, how it was trained, and on what."""
    from eve.model.checkpoint import model_file

    trained = [f"{meta['step']:,} steps" if "step" in meta else "? steps", f"from {meta['init']}"]
    items = [
        ("model", str(model_file(args.model))),
        ("trained", " ".join(trained)),
        ("sets", f"{args.evals} · {', '.join(args.sets)}"),
        ("device", str(device)),
    ]
    return panel("eve evaluate", items)


def _score_cell(scores: dict[str, float], key: str) -> str:
    """A recall per intent as ``96%``, the summary scores as ``53.3%``, as in the training log.
    Blank when the set has no examples of that intent."""
    if key not in scores:
        return ""
    return f"{scores[key]:.0%}" if key.startswith("recall_") else f"{scores[key]:.1%}"


def _scores_table(results: dict[str, "SetResult"]) -> "Table":
    """One row per scored set: its size, the recall per intent and their mean, tools and exact,
    then a row per group of its rows, if it has groups. A set without examples of an intent has
    a blank recall for it."""
    out = table("set", "n", *(label for label, _ in SCORE_COLUMNS), first_style="bold")
    for name, result in results.items():
        cells = [_score_cell(result.scores, key) for _, key in SCORE_COLUMNS]
        out.add_row(name, f"{result.examples:,}", *cells)
        for group, (n, scores) in result.groups.items():  # novel-tools: shared word or not
            cells = [_score_cell(scores, key) for _, key in SCORE_COLUMNS]
            out.add_row(Text(f"  {group}", style="dim"), f"{n:,}", *cells)
    return out


def _confusion_table(name: str, result: "SetResult") -> "Table":
    """A row per gold intent the set has, a column per intent answered; right answers green."""
    out = table(f"{name}: gold \\ answered", *INTENTS, first_style="dim")
    for gold, said in result.confusion.items():
        if sum(said.values()):
            styles = ("bold green" if s == gold else "dim" * (not n) for s, n in said.items())
            out.add_row(
                gold,
                *(Text(f"{n:,}", style) for n, style in zip(said.values(), styles, strict=True)),
            )
    return out


def _answered(counts: dict[str, int]) -> Text:
    """``call 12 · unsup 3``: how often each intent was answered, most often first."""
    ranked = sorted(((n, intent) for intent, n in counts.items() if n), reverse=True)
    return Text(" · ").join(Text(f"{intent[:5]} {n:,}") for n, intent in ranked)


def _tools_table(name: str, result: "SetResult") -> "Table":
    """A row per gold tool, worst exact accuracy first, with what its examples were answered."""
    out = table("gold tool", "n", "intent", "tools", "exact", "values", "full", "answered")
    out.title, out.title_justify, out.title_style = f"{name} by tool", "left", "bold"
    out.columns[-1].justify, out.columns[-1].no_wrap = "left", False
    for tool in result.by_tool:
        accuracies = (
            tool.intent_acc,
            tool.tools_acc,
            tool.exact_acc,
            tool.value_acc,
            tool.full_acc,
        )
        cells = [f"{tool.examples:,}", *("" if a is None else f"{a:.0%}" for a in accuracies)]
        out.add_row(tool.tool, *cells, _answered(tool.answered))
    return out


def _print_evaluation(results: dict[str, "SetResult"], as_json: bool, by_tool: bool) -> None:
    """The scores table and a confusion table per set (and a per-tool table with ``by_tool``),
    or all of it as one JSON object."""
    if as_json:
        print(json.dumps({name: result.to_json(by_tool) for name, result in results.items()}))
        return
    console.print(_scores_table(results))
    for name, result in results.items():
        console.print(_confusion_table(name, result))
        if by_tool:
            console.print(_tools_table(name, result))


def evaluate(args: argparse.Namespace) -> None:
    """Score a fine-tuned model on whole task files: recall per intent, tools and exact."""
    from eve import progress
    from eve.data.generate.build import Sources
    from eve.evaluate import evaluate_set, load_questions_model, set_files

    device = pick_device(args.device)
    try:
        paths = set_files(args.evals, args.sets)
        ckpt = load_questions_model(args.model, device)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not args.json:
        console.print(_evaluate_panel(args, ckpt.meta, device))
    library, results = Sources.load().library, {}
    with progress.bar(len(paths), "evaluate", "sets", leave=False) as bar:
        for name, path in paths.items():
            bar.set_stats(name)
            results[name] = evaluate_set(ckpt, path, library, args.batch_size)
            bar.update()
    _print_evaluation(results, args.json, args.by_tool)


def _calibration_table(
    name: str, counts: dict[str, int], before: dict[str, Any], after: dict[str, Any]
) -> "Table":
    """A row per answer: how many, their accuracy, and the mean confidence and expected
    calibration error before and after (``eve.calibrate.Reliability``)."""
    out = table(name, "n", "accuracy", "confidence", "", "ECE", "", first_style="bold")
    out.title = f"{counts['familiar']:,} familiar · {counts['new']:,} new"
    out.title_justify, out.title_style = "left", "dim"
    for head, old in before.items():
        new = after[head]
        out.add_row(
            head,
            f"{old.n:,}",
            f"{old.accuracy:.1%}",
            Text(f"{old.mean_confidence:.1%} →", style="dim"),
            f"{new.mean_confidence:.1%}",
            Text(f"{old.ece():.3f} →", style="dim"),
            Text(f"{new.ece():.3f}", style="bold green" if new.ece() < old.ece() else "bold red"),
        )
    return out


def _fit_and_check(
    ckpt: "Checkpoint[Eve]",
    paths: dict[str, Path],
    sets: list[str],
    fit_on: dict[str, str],
    batch_size: int,
) -> tuple["Calibrations", dict[str, "TaskSet"], dict[str, list[dict[str, "Reliability"]]]]:
    """Fit the familiar and the new temperatures on their sets, then measure every set's
    reliability before (``known`` only) and after; one progress bar for all the passes."""
    from eve import progress
    from eve.calibrate import REGIMES, fit, known_tools, load_set, reliability
    from eve.data.generate.build import Sources
    from eve.infer.predict import Calibrations

    library = Sources.load().library
    with progress.bar(len(REGIMES) + len(sets), "calibrate", "passes", leave=False) as bar:
        bar.set_stats("reading the training tools")
        known = Calibrations(known=known_tools(paths["train"], library))
        tasks = {name: load_set(ckpt, paths[name], library, batch_size, known) for name in sets}
        fitted = {}
        for regime, name in fit_on.items():
            bar.set_stats(f"fitting {regime} on {name}")
            fitted[regime] = fit(ckpt.model, tasks[name].batches, tasks[name].only(regime))
            bar.update()
        cal = Calibrations(fitted["familiar"], fitted["new"], known.known)
        results = {}
        for name in sets:
            bar.set_stats(name)
            results[name] = reliability(ckpt.model, tasks[name], [known, cal])
            bar.update()
    return cal, tasks, results


def _calibration_json(
    cal: "Calibrations",
    fit_on: dict[str, str],
    tasks: dict[str, "TaskSet"],
    results: dict[str, list[dict[str, "Reliability"]]],
) -> dict[str, Any]:
    """``eve calibrate --json``: the temperatures, the sets they were fitted on, and each set's
    reliability per answer before and after."""
    from eve.calibrate import HEADS

    sets_json = {
        name: {
            "examples": tasks[name].counts(),
            **{h: {"before": old[h].to_json(), "after": new[h].to_json()} for h in HEADS},
        }
        for name, (old, new) in results.items()
    }
    temperatures = {"familiar": cal.familiar.to_dict(), "new": cal.new.to_dict()}
    return {"calibration": temperatures, "fitted_on": fit_on, "sets": sets_json}


def _calibration_panel(
    model: str, cal: "Calibrations", fit_on: dict[str, str], train_path: Path
) -> "Panel":
    """The fitted temperatures per regime and where the known tools came from."""
    from eve.calibrate import HEADS, REGIMES

    items: list[tuple[str, str | Text]] = [("model", model)]
    for regime in REGIMES:
        temps = getattr(cal, regime)
        line = " · ".join(f"{head} {getattr(temps, head):.2f}" for head in HEADS)
        items.append((regime, f"{line}   fitted on {fit_on[regime]}"))
    items.append(("known tools", f"{len(cal.known):,} schemas in {train_path}"))
    return panel("eve calibrate", items)


def calibrate(args: argparse.Namespace) -> None:
    """Fit temperatures for familiar and for new tools and show how calibrated the
    probabilities are before and after on each set; ``--save`` writes them into the model."""
    from eve.calibrate import save_calibration
    from eve.evaluate import load_questions_model, set_files
    from eve.model.checkpoint import model_file

    device = pick_device(args.device)
    fit_on = {"familiar": args.fit_familiar, "new": args.fit_new}
    sets = list(dict.fromkeys([*fit_on.values(), *args.sets]))  # the fit sets first, once
    try:
        paths = set_files(args.evals, sets) | set_files(args.task, ["train"])
        ckpt = load_questions_model(args.model, device)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not ckpt.meta.get("joint"):
        raise SystemExit(f"{args.model} is not from a joint run: its decoder writes no values")
    cal, tasks, results = _fit_and_check(ckpt, paths, sets, fit_on, args.batch_size)
    if args.save:
        save_calibration(model_file(args.model), cal, fit_on)

    if args.json:
        print(json.dumps(_calibration_json(cal, fit_on, tasks, results)))
        return
    console.print(_calibration_panel(str(args.model), cal, fit_on, paths["train"]))
    for name, (old, new) in results.items():
        console.print(_calibration_table(name, tasks[name].counts(), old, new))
    if args.save:
        success("temperatures saved", model_file(args.model))
    else:
        console.print(Text("not saved: add --save to write them into the model", style="dim"))


def add_ask(commands: Subcommands) -> None:
    """Register ``eve ask``."""
    parser = commands.add_parser("ask", help="ask the model about a request")
    parser.set_defaults(run=ask)
    parser.add_argument("text", nargs="*", help="requests; none: type them one by one")
    parser.add_argument("--tools", help="tool names from data/tools: set_lights,set_thermostat")
    parser.add_argument(
        "--schema", action="append", help="a JSON file of tool schemas (repeatable)"
    )
    parser.add_argument("--model", default=defaults.FINETUNED_MODEL)
    add_json_arg(parser, "one JSON line per request")
    add_device_arg(parser)


def add_evaluate(commands: Subcommands) -> None:
    """Register ``eve evaluate``."""
    from eve.evaluate import SETS

    parser = commands.add_parser("evaluate", help="score a fine-tuned model on whole eval sets")
    parser.set_defaults(run=evaluate)
    add_model_arg(parser)
    parser.add_argument("--evals", default=defaults.EVALS_DIR, help="where the sets are")
    parser.add_argument(
        "--sets", nargs="+", default=list(SETS), help="set names, e.g. probe-candidates"
    )
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    parser.add_argument("--by-tool", action="store_true", help="also a row per gold tool")
    add_json_arg(parser, "the results as one JSON object")
    add_device_arg(parser)


def add_calibrate(commands: Subcommands) -> None:
    """Register ``eve calibrate``."""
    from eve.evaluate import SETS

    parser = commands.add_parser("calibrate", help="fit the probabilities' temperatures")
    parser.set_defaults(run=calibrate)
    add_model_arg(parser)
    parser.add_argument("--task", default=defaults.TASK_DIR, help="where train.jsonl is")
    parser.add_argument("--evals", default=defaults.EVALS_DIR, help="where the sets are")
    parser.add_argument(
        "--fit-familiar", default="val", help="set whose familiar-tool examples fit those"
    )
    parser.add_argument(
        "--fit-new", default="probe-candidates", help="set whose new-tool examples fit those"
    )
    parser.add_argument("--sets", nargs="+", default=list(SETS), help="sets to check them on")
    parser.add_argument("--save", action="store_true", help="write them into the model file")
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    add_json_arg(parser, "the results as one JSON object")
    add_device_arg(parser)


def add_fill(commands: Subcommands) -> None:
    """Register ``eve fill``."""
    parser = commands.add_parser("fill", help="fill <x0> holes with a pretrained model")
    parser.set_defaults(run=fill)
    parser.add_argument("text", nargs="*", help='e.g. "Paris is the capital of <x0>."')
    parser.add_argument("--model", default=defaults.PRETRAINED_MODEL)
    parser.add_argument(
        "--max-new", type=int, default=defaults.MAX_NEW_TOKENS, help="most tokens to write"
    )
    add_device_arg(parser)
