import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

from colibri.config import PRESETS
from colibri.runtime import pick_device

DEFAULT_PARQUET = "data/raw/fineweb-edu/*.parquet"
DEFAULT_TOKENIZER = "artifacts/tokenizer/colibri-8k.json"


def _count(text: str) -> int:
    """'250M' -> 250_000_000, '2k' -> 2000, '1000' -> 1000."""
    scale = {"k": 10**3, "m": 10**6, "b": 10**9}.get(text[-1].lower(), 1)
    return int(float(text[:-1] if scale > 1 else text) * scale)


def _parquet_files(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no parquet files match {pattern}")
    return paths


def info(_: argparse.Namespace) -> None:
    print(f"torch {torch.__version__}, device: {pick_device()}")
    print(f"{'preset':<8}{'d_model':>9}{'heads':>7}{'enc':>5}{'dec':>5}{'q':>4}{'params':>10}")
    for name, cfg in PRESETS.items():
        print(
            f"{name:<8}{cfg.d_model:>9}{cfg.n_heads:>7}{cfg.n_encoder_layers:>5}"
            f"{cfg.n_decoder_layers:>5}{cfg.n_question_layers:>4}{cfg.approx_params() / 1e6:>9.1f}M"
        )


def model(args: argparse.Namespace) -> None:
    """Learn (M3): parameters per part and memory size, like the book's count in ch. 4.6."""
    from colibri.config import preset
    from colibri.model.colibri import Colibri

    names = [args.preset] if args.preset else list(PRESETS)
    counts = {name: Colibri(preset(name)).parameter_counts() for name in names}
    print(f"{'':<12}" + "".join(f"{name:>12}" for name in names))
    for part in counts[names[0]]:
        print(f"{part:<12}" + "".join(f"{counts[n][part]:>12,}" for n in names))
    # float32: 4 bytes per parameter. The LM head adds nothing: it reuses the token embeddings.
    print(f"{'MB (fp32)':<12}" + "".join(f"{counts[n]['total'] * 4 / 2**20:>12.1f}" for n in names))


def tokenizer_train(args: argparse.Namespace) -> None:
    from colibri.data.corpus import iter_documents, take_bytes, tokenizer_texts
    from colibri.tokenizer.bpe import BPETokenizer

    corpus = take_bytes(iter_documents(_parquet_files(args.parquet)), _count(args.text))
    start = time.time()
    tokenizer = BPETokenizer.train(
        tokenizer_texts(corpus, args.tools, args.commands, args.weight),
        vocab_size=args.vocab_size,
        progress=True,
    )
    tokenizer.save(args.out)
    print(f"{tokenizer.vocab_size} tokens in {time.time() - start:.0f}s -> {args.out}")


def tokenizer_stats(args: argparse.Namespace) -> None:
    """Learn (M1): tokens per word on unseen web text, on commands and on tool schemas."""
    from colibri.data.corpus import iter_documents, take_bytes
    from colibri.data.schema import load_library
    from colibri.tokenizer.bpe import BPETokenizer

    ours = BPETokenizer.load(args.tokenizer)
    docs = list(
        take_bytes(
            iter_documents(_parquet_files(args.parquet)), _count(args.text), _count(args.skip)
        )
    )
    tools = load_library(args.tools)
    commands = [
        line.strip()
        for line in Path(args.commands).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    schemas = [t.render_parts() for t in tools.values()]
    samples = {
        "web text (held out)": docs,
        "commands": commands,
        "tool schemas": schemas,
    }

    # A schema is a list of parts (see Tool.render_parts); other samples are plain strings.
    def ours_encode(t: str | list[str]) -> list[int]:
        parts = [t] if isinstance(t, str) else t
        return [i for part in parts for i in ours.encode(part, specials=True)]

    print(f"{'':<22}{'words':>9}{'tokens/word':>13}")
    for name, texts in samples.items():
        words = sum(len(" ".join(t if isinstance(t, list) else [t]).split()) for t in texts)
        n_ours = sum(len(ours_encode(t)) for t in texts)
        print(f"{name:<22}{words:>9,}{n_ours / words:>13.2f}")
    print(f"vocabulary: {ours.vocab_size:,}")

    for text in [commands[1], schemas[0]]:
        print("\n" + " | ".join(ours.pieces(ours_encode(text))))


def corpus(args: argparse.Namespace) -> None:
    from colibri.data.corpus import encode_corpus

    meta = encode_corpus(
        _parquet_files(args.parquet),
        args.tokenizer,
        args.out,
        train_tokens=_count(args.train_tokens),
        val_tokens=_count(args.val_tokens),
        workers=args.workers,
    )
    print(f"train {meta['train_tokens']:,} and val {meta['val_tokens']:,} tokens -> {args.out}")


def pretrain(args: argparse.Namespace) -> None:
    from colibri.config import preset
    from colibri.progress import style
    from colibri.train.loop import TrainConfig
    from colibri.train.pretrain import pretrain as run
    from colibri.train.pretrain import steps_for

    overrides = {"rope": args.rope, "qk_norm": args.qk_norm}
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    cfg = preset(args.preset, **overrides)
    steps = args.steps or steps_for(_count(args.tokens), args.batch_size, cfg.max_encoder_len)
    train_cfg = TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=min(args.warmup, steps),
        dtype=args.dtype,
        seed=args.seed,
        eval_every=args.eval_every,
        save_every=args.eval_every,
        log_every=args.log_every,
    )
    device = pick_device(args.device)
    print(
        f"{style('pretrain', '1')} {args.preset} · {cfg.approx_params() / 1e6:.1f}M params · "
        f"{device} · {args.dtype} · lr {args.lr:g} · dropout {cfg.dropout:g} · "
        + "".join(f"{name} · " for name in ("rope", "qk_norm") if getattr(cfg, name))
        + f"{steps:,} steps × {args.batch_size} windows of {cfg.max_encoder_len} tokens"
        + (f" · stopping at step {args.stop_after:,}" if args.stop_after else "")
    )
    run(
        cfg,
        train_cfg,
        args.data,
        args.tokenizer,
        args.out,
        device,
        resume=args.resume,
        stop_after=args.stop_after,
    )


def finetune(args: argparse.Namespace) -> None:
    from colibri.config import preset
    from colibri.data.generate.build import Sources, read_jsonl
    from colibri.progress import style
    from colibri.train.finetune import finetune as run
    from colibri.train.loop import TrainConfig

    n_train = len(read_jsonl(Path(args.task) / "train.jsonl"))
    steps = args.steps or round(args.epochs * n_train / args.batch_size)
    lr = args.lr or (1e-3 if args.freeze else 3e-4)
    train_cfg = TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        lr=lr,
        warmup_steps=min(args.warmup, steps),
        dtype=args.dtype,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_batches=10_000,  # all of val: it is small
        save_every=args.eval_every,
        log_every=args.log_every,
    )
    device = pick_device(args.device)
    start = args.init or f"{args.scratch} from scratch"
    print(
        f"{style('finetune', '1')} questions · from {start} · {device} · lr {lr:g} · "
        + ("body frozen · " if args.freeze else "")
        + f"{steps:,} steps × {args.batch_size} ({steps * args.batch_size / n_train:.1f} epochs)"
        + (f" · stopping at step {args.stop_after:,}" if args.stop_after else "")
    )
    run(
        train_cfg,
        args.task,
        args.out,
        device,
        Sources.load().library,
        init=args.init,
        scratch=preset(args.scratch) if args.scratch else None,
        tokenizer_path=args.tokenizer,
        freeze=args.freeze,
        overrides={} if args.dropout is None else {"dropout": args.dropout},
        resume=args.resume,
        stop_after=args.stop_after,
    )


def compare(args: argparse.Namespace) -> None:
    """Learn (M4): val metrics of several runs side by side, one row per eval step."""
    runs = {}
    for run in args.runs:
        path = Path(run) / "metrics.jsonl"
        if not path.exists():
            raise SystemExit(f"no metrics in {run}")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        start = next((r for r in records if r["kind"] == "start"), {})
        vals = {r["step"]: r for r in records if r["kind"] == "val"}
        runs[Path(run).name] = (start, vals)

    width = max(12, *(len(name) for name in runs)) + 2
    print(f"{'':<14}" + "".join(f"{name:>{width}}" for name in runs))
    for label, get in [
        ("lr", lambda s: s.get("train_config", {}).get("lr")),
        ("dropout", lambda s: s.get("model_config", {}).get("dropout")),
        ("rope", lambda s: s.get("model_config", {}).get("rope", False)),
        ("qk_norm", lambda s: s.get("model_config", {}).get("qk_norm", False)),
    ]:
        print(f"{label:<14}" + "".join(f"{get(s)!s:>{width}}" for s, _ in runs.values()))
    steps = sorted({step for _, vals in runs.values() for step in vals})
    for metric in args.metric:
        print(f"\n{metric}")
        for step in steps:
            cells = []
            for _, vals in runs.values():
                v = vals.get(step, {}).get(metric)
                cells.append(
                    "" if v is None else f"{v:.1%}" if metric.endswith("acc") else f"{v:.4f}"
                )
            print(f"  step {step:<8,}" + "".join(f"{c:>{width}}" for c in cells))


def fill(args: argparse.Namespace) -> None:
    """Learn (M4): ask a pretrained model to fill the <x0>, <x1>… holes in a sentence."""
    from colibri import progress
    from colibri.model.checkpoint import load_checkpoint
    from colibri.train.pretrain import SAMPLES, fill_in

    texts = args.text or list(SAMPLES)
    if bad := [t for t in texts if "<x0>" not in t]:
        raise SystemExit(f"put a hole like <x0> in the text: {bad[0]!r}")
    device = pick_device(args.device)
    with progress.bar(len(texts), "fill", "prompts", leave=False) as bar:
        bar.set_stats(f"loading {Path(args.model).name}")
        ckpt = load_checkpoint(args.model, device)
        filled = []
        for text in texts:
            bar.set_stats("decoding")
            filled.append(fill_in(ckpt.model, ckpt.tokenizer, text, args.max_new, progress.marks()))
            bar.update()

    def row(label: str, value: str) -> None:
        print(f"  {progress.style(f'{label:<11}', '2', stream=sys.stdout)}{value}")

    meta = ckpt.meta
    trained = f"{meta['step']:,} steps" if "step" in meta else "?"
    if "val_lm" in meta:
        trained += f" · val loss {meta['val_lm']:.3f}"
    print()
    row("checkpoint", str(args.model))
    row("trained", trained)
    row("device", str(device))
    print()
    for i, line in enumerate(filled, 1):
        print(f"  {progress.style(f'{i:>2}', '2', stream=sys.stdout)}  {line}")
    print()


def _tools(args: argparse.Namespace) -> list:
    """The tools for ``colibri ask``: names from the library and/or schemas from JSON files."""
    from colibri.data.generate.build import Sources
    from colibri.data.schema import GENERATION_KEYS, Tool

    tools = []
    if args.tools:
        library = Sources.load().library
        names = [name.strip() for name in args.tools.split(",") if name.strip()]
        if unknown := [name for name in names if name not in library]:
            raise SystemExit(f"unknown tools {unknown}; see data/tools/*.json or use --schema")
        tools += [library[name] for name in names]
    for path in args.schema or []:
        data = json.loads(Path(path).read_text())
        rows = data["tools"] if isinstance(data, dict) else data  # a data/tools file, or a list
        tools += [
            Tool.from_dict({k: v for k, v in t.items() if k not in GENERATION_KEYS}) for t in rows
        ]
    if not tools:
        raise SystemExit("give the tools with --tools a,b,c and/or --schema FILE")
    return tools


def ask(args: argparse.Namespace) -> None:
    """Learn (M7): what the question heads answer for a typed request, with probabilities."""
    from colibri.infer.predict import STOP, answer_questions
    from colibri.model.checkpoint import load_checkpoint
    from colibri.progress import style

    tools = _tools(args)
    ckpt = load_checkpoint(args.model, pick_device(args.device))

    def show(text: str) -> None:
        a = answer_questions(ckpt.model, ckpt.tokenizer, text, tools)
        if args.json:
            print(json.dumps({"utterance": text, **a.to_json()}))
            return

        def ranked(probs: dict[str, float], n: int) -> str:
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:n]
            return " · ".join(
                (style(f"{k} {p:.0%}", "1", stream=sys.stdout) if i == 0 else f"{k} {p:.0%}")
                for i, (k, p) in enumerate(top)
            )

        print(f"  {style(f'{"intent":<8}', '2', stream=sys.stdout)} {ranked(a.intent, 4)}")
        for i, slot in enumerate(a.slots, 1):
            print(f"  {style(f'{f"call {i}":<8}', '2', stream=sys.stdout)} {ranked(slot, 3)}")
            if max(slot, key=slot.get) == STOP:
                break

    if args.text:
        for text in args.text:
            if not args.json:
                print(text)
            show(text)
        return
    names = ", ".join(t.name for t in tools)
    print(f"{args.model} · tools: {names}\ntype a request (empty line or Ctrl-D to quit)")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        show(text)


def data_build(args: argparse.Namespace) -> None:
    from colibri.config import preset
    from colibri.data.generate.build import build
    from colibri.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(args.tokenizer) if Path(args.tokenizer).exists() else None
    if tokenizer is None:
        print(f"no tokenizer at {args.tokenizer}: not checking example lengths")
    report = build(
        args.out,
        _count(args.templates),
        _count(args.heldout),
        args.seed,
        tokenizer,
        preset("small"),
    )
    for name, stats in report["splits"].items():
        print(f"{name:<18} {stats['examples']:>7,}  {stats['intent']}")
    print(f"rejected chat lines: {report['rejected_chat_lines']} (see `colibri data check`)")
    print(f"report -> {args.out}/report.json")


def data_external(args: argparse.Namespace) -> None:
    from colibri.data.generate.external import convert_external

    report = convert_external(args.raw, args.out, args.seed)
    print(json.dumps(report, indent=2))
    if not report:
        print(f"nothing found in {args.raw}/xlam or {args.raw}/massive")


def data_batch(args: argparse.Namespace) -> None:
    from colibri.data.generate.build import Sources, make_batch, next_batch_name

    batches = Path(args.dir) / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    for _ in range(args.count):
        name = next_batch_name(batches)
        seed = int(name.split("-")[1])
        sources = Sources.load(generation_dir=args.dir)
        batch = make_batch(sources, name, args.n, args.split, seed, args.ask)
        (batches / f"{name}.json").write_text(json.dumps(batch, indent=1, ensure_ascii=False))
        print(f"{batches / name}.json: {args.n} {args.split} scenarios")


def data_check(args: argparse.Namespace) -> None:
    from colibri.data.generate.build import Sources, check_written, read_jsonl

    batch = json.loads((Path(args.dir) / "batches" / f"{args.batch}.json").read_text())
    written_path = Path(args.dir) / "written" / f"{args.batch}.jsonl"
    written = read_jsonl(written_path) if written_path.exists() else []
    examples, rejects = check_written(batch, written, Sources.load(generation_dir=args.dir).library)
    missing = {s["id"] for s in batch["scenarios"]} - {line.get("id") for line in written}
    for line, errors in rejects:
        print(f"{line.get('id')}: {line.get('utterance')!r}\n    " + "\n    ".join(errors))
    print(f"{len(examples)} ok, {len(rejects)} rejected, {len(missing)} scenarios not written")
    if rejects or missing:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="colibri")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="show the device and model presets").set_defaults(run=info)
    p = commands.add_parser("model", help="parameters per part of a preset (M3)")
    p.set_defaults(run=model)
    p.add_argument("--preset", choices=sorted(PRESETS), default=None, help="default: all")

    tok = commands.add_parser("tokenizer", help="train or inspect the BPE tokenizer (M1)")
    tok_commands = tok.add_subparsers(dest="tokenizer_command", required=True)
    for name, run, help_ in [
        ("train", tokenizer_train, "train the BPE on web text, tool schemas and commands"),
        ("stats", tokenizer_stats, "tokens per word and sample splits"),
    ]:
        p = tok_commands.add_parser(name, help=help_)
        p.set_defaults(run=run)
        p.add_argument("--parquet", default=DEFAULT_PARQUET)
        p.add_argument("--tools", default="data/tools")
        p.add_argument("--commands", default="data/tokenizer/commands.txt")
        if name == "train":
            p.add_argument("--text", default="50M", help="bytes of web text to train on")
            p.add_argument("--vocab-size", type=int, default=8192)
            p.add_argument("--weight", type=int, default=50, help="weight of schemas and commands")
            p.add_argument("--out", default=DEFAULT_TOKENIZER)
        else:
            p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
            p.add_argument("--text", default="5M", help="bytes of web text to measure on")
            p.add_argument("--skip", default="100M", help="bytes to skip (the training text)")

    p = commands.add_parser("corpus", help="encode the pretraining corpus to token ids (M1)")
    p.set_defaults(run=corpus)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--out", default="data/bin")
    p.add_argument("--train-tokens", default="250M")
    p.add_argument("--val-tokens", default="2M")
    p.add_argument("--workers", type=int, default=None)

    p = commands.add_parser("pretrain", help="span-corruption pretraining (M4)")
    p.set_defaults(run=pretrain)
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--data", default="data/bin")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--out", default="artifacts/runs/pretrain-small")
    p.add_argument("--tokens", default="250M", help="corpus tokens to read (sets --steps)")
    p.add_argument("--steps", type=int, default=None, help="default: from --tokens")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--dropout", type=float, default=None, help="default: the preset's (0.1)")
    p.add_argument("--rope", action="store_true", help="rotary positions, no position tables")
    p.add_argument("--qk-norm", action="store_true", help="RMS-normalize queries and keys")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=500, help="also how often to save")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--stop-after", type=int, default=None, help="stop early at this step")

    p = commands.add_parser("finetune", help="fine-tune the question heads (M7)")
    p.set_defaults(run=finetune)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", help="a pretraining checkpoint or an earlier finetune's model.pt")
    start.add_argument("--scratch", choices=sorted(PRESETS), help="random init (experiment E1)")
    p.add_argument("--freeze", action="store_true", help="train only the question block (E2)")
    p.add_argument("--task", default="data/task")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="for --scratch")
    p.add_argument("--out", default="artifacts/runs/questions")
    p.add_argument("--epochs", type=float, default=2.0, help="sets --steps")
    p.add_argument("--steps", type=int, default=None, help="default: from --epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 frozen, else 3e-4")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--dropout", type=float, default=None, help="default: the checkpoint's")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=500, help="also how often to save")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    p.add_argument("--stop-after", type=int, default=None, help="stop early at this step")
    p = commands.add_parser("ask", help="ask the question heads about a request (M7)")
    p.set_defaults(run=ask)
    p.add_argument("text", nargs="*", help="requests; none: type them one by one")
    p.add_argument("--tools", help="tool names from data/tools: set_lights,set_thermostat")
    p.add_argument("--schema", action="append", help="a JSON file of tool schemas (repeatable)")
    p.add_argument("--model", default="artifacts/runs/questions-full/model.pt")
    p.add_argument("--json", action="store_true", help="one JSON line per request")
    p.add_argument("--device", default=None)

    p = commands.add_parser("compare", help="val metrics of training runs side by side (M4)")
    p.set_defaults(run=compare)
    p.add_argument("runs", nargs="+", help="run directories, e.g. artifacts/runs/ab-*")
    p.add_argument("--metric", nargs="+", default=["val_content", "val_content_acc", "val_lm"])

    p = commands.add_parser("fill", help="fill <x0> holes with a pretrained model (M4)")
    p.set_defaults(run=fill)
    p.add_argument("text", nargs="*", help='e.g. "Paris is the capital of <x0>."')
    p.add_argument("--model", default="artifacts/runs/pretrain-small/model.pt")
    p.add_argument("--max-new", type=int, default=24, help="most tokens to write")
    p.add_argument("--device", default=None)

    data = commands.add_parser("data", help="build the task dataset (M6)")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    p = data_commands.add_parser("build", help="templates + chat-written -> data/task/*.jsonl")
    p.set_defaults(run=data_build)
    p.add_argument("--out", default="data/task")
    p.add_argument("--templates", default="50k", help="template examples for train")
    p.add_argument("--heldout", default="2k", help="template examples with held-out tools")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p = data_commands.add_parser("external", help="convert xLAM and MASSIVE -> data/external")
    p.set_defaults(run=data_external)
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/external")
    p.add_argument("--seed", type=int, default=0)
    p = data_commands.add_parser("batch", help="new scenario batches for chat writers")
    p.set_defaults(run=data_batch)
    p.add_argument("--split", choices=["train", "val", "probe"], default="train")
    p.add_argument("--n", type=int, default=50, help="scenarios per batch")
    p.add_argument("--count", type=int, default=1, help="how many batches")
    p.add_argument("--ask", type=float, default=None, help="share of ASK scenarios, e.g. 0.3")
    p.add_argument("--dir", default="data/generation")
    p = data_commands.add_parser("check", help="validate a chat-written batch")
    p.set_defaults(run=data_check)
    p.add_argument("batch", help="e.g. batch-0001")
    p.add_argument("--dir", default="data/generation")

    args = parser.parse_args(argv)
    args.run(args)
