"""Commands about a model to ship: ``info``, ``export`` and ``retriever eval``."""

import argparse
import json

import torch
from rich.text import Text

from eve import defaults
from eve.cli.common import Subcommands, add_device_arg, add_json_arg
from eve.config import PRESETS
from eve.progress import console, format_metric, panel, rows, success, table
from eve.runtime import pick_device


def info(args: argparse.Namespace) -> None:
    """Print the torch version, the device, each preset's size, and (M3) its parameters per
    part and memory size, like the book's count in ch. 4.6."""
    from eve.config import preset
    from eve.model.eve import Eve

    console.print(rows([("torch", torch.__version__), ("device", str(pick_device()))]))
    console.print()
    presets = table("preset", "d_model", "heads", "encoder", "decoder", "questions", "params")
    presets.columns[0].style = "bold cyan"
    for name, cfg in PRESETS.items():
        sizes = (cfg.d_model, cfg.n_heads, cfg.n_encoder_layers, cfg.n_decoder_layers)
        presets.add_row(
            name,
            *(str(size) for size in sizes),
            str(cfg.n_question_layers),
            f"{cfg.approx_params() / 1e6:.1f}M",
        )
    console.print(presets)
    console.print()

    names = [args.preset] if args.preset else list(PRESETS)
    counts = {name: Eve(preset(name)).parameter_counts() for name in names}
    parts = table("", *names, first_style="dim")
    for part in counts[names[0]]:
        if part == "total":
            parts.add_section()
        cells = [f"{counts[n][part]:,}" for n in names]
        parts.add_row(part, *cells, style="bold" if part == "total" else "")
    # float32: 4 bytes per parameter. The LM head adds nothing: it reuses the token embeddings.
    parts.add_row("MB (fp32)", *(f"{counts[n]['total'] * 4 / 2**20:.1f}" for n in names))
    console.print(parts)


def export(args: argparse.Namespace) -> None:
    """Publish (M10): write a fine-tuned model and its model card for a Hugging Face Hub repo."""
    from eve.export import export_model
    from eve.model.checkpoint import MODEL_FILE, RETRIEVER_FILE

    try:
        out = export_model(args.model, args.out, args.dtype, args.repo, args.retriever)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    size = (out / MODEL_FILE).stat().st_size / 1e6
    extra = f" · {RETRIEVER_FILE}" if args.retriever else ""
    success(f"{MODEL_FILE} ({size:.0f} MB, {args.dtype}){extra} and README.md", out)
    console.print(Text(f"upload: hf upload {args.repo} {out} .", style="dim"))


def retriever_eval(args: argparse.Namespace) -> None:
    """Retrieve (M11): how often each scorer puts every correct tool in the top 5 of a
    registry: BM25, and with a retriever checkpoint, the retriever (dense). Registries are
    filled with random tools, and with ``similar`` ones (the hard case)."""
    from eve.data.generate.build import Sources
    from eve.infer.search import BM25, Dense, Scorer
    from eve.model.checkpoint import load_checkpoint
    from eve.tasks.retrieval import TOP, evaluate_scorers, load_splits, suite

    scorers: dict[str, Scorer] = {"bm25": BM25()}
    if args.retriever:
        ckpt = load_checkpoint(args.retriever, pick_device(args.device))
        scorers["dense"] = Dense(ckpt.model, ckpt.tokenizer)  # type: ignore[arg-type]
    examples = load_splits(Sources.load().library, args.evals, args.splits)
    suites = suite(examples, args.sizes, fills=args.fill)
    results = evaluate_scorers(scorers, suites)
    if args.json:
        print(json.dumps(results, indent=1))
        return
    sizes = " · ".join(f"{key} {len(cases):,}" for key, cases in suites.items())
    console.print(
        panel("eve retriever eval", [("requests", sizes), ("retriever", args.retriever or "–")])
    )
    hard = [key for key in suites if " " in key]  # "val@25 similar"
    for keys in ([key for key in suites if key not in hard], hard):
        for metric in (f"all@{TOP}", f"recall@{TOP}", "mrr") if keys else ():
            out = table(metric, *keys, first_style="bold")
            for name, by_suite in results.items():
                out.add_row(name, *(format_metric(metric, by_suite[key][metric]) for key in keys))
            console.print(out)


def add_info(commands: Subcommands) -> None:
    parser = commands.add_parser("info", help="the device, and the model presets' sizes (M3)")
    parser.set_defaults(run=info)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="parameters per part of only this one",
    )


def add_export(commands: Subcommands) -> None:
    from eve.export import DTYPES as EXPORT_DTYPES

    parser = commands.add_parser("export", help="write a fine-tuned model for the Hub (M10)")
    parser.set_defaults(run=export)
    parser.add_argument("model", nargs="?", default=defaults.FINETUNED_MODEL)
    parser.add_argument("--out", default="dist/model", help="the directory to write")
    parser.add_argument("--dtype", choices=list(EXPORT_DTYPES), default="float16")
    parser.add_argument("--repo", default="<user>/<repo>", help="the Hub repo, for the card")
    parser.add_argument("--retriever", default=None, help="a retriever model.pt to ship with it")


def add_retriever(commands: Subcommands) -> None:
    group = commands.add_parser("retriever", help="score tool retrieval (M11)")
    retriever_commands = group.add_subparsers(dest="retriever_command", required=True)
    parser = retriever_commands.add_parser("eval", help="top-5 recall of BM25 and a retriever")
    parser.set_defaults(run=retriever_eval)
    parser.add_argument("retriever", nargs="?", help="a retriever model.pt; BM25 alone without")
    parser.add_argument("--evals", default=defaults.EVALS_DIR)
    parser.add_argument("--splits", nargs="+", default=["val", "heldout", "massive", "novel"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 50], help="registry sizes")
    parser.add_argument(
        "--fill",
        nargs="+",
        choices=["random", "similar"],
        default=["random", "similar"],
        help="what fills a registry up: random tools, or ones like the correct tools",
    )
    add_json_arg(parser, "the results as one JSON object")
    add_device_arg(parser)
