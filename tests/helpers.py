"""Builders that several test modules share."""

import json
from pathlib import Path
from typing import Any

import torch

from eve import defaults
from eve.config import ModelConfig, preset
from eve.data.render import Call, TaskExample
from eve.data.schema import Tool
from eve.experiment import Experiment, run
from eve.model.checkpoint import save_checkpoint
from eve.model.eve import Eve
from eve.model.retriever import Retriever, RetrieverConfig
from eve.tasks.questions import TASK
from eve.tokenizer.bpe import BPETokenizer
from eve.train.config import TrainConfig


def examples(library: dict[str, Tool]) -> list[TaskExample]:
    """Six labelled requests, one or more of each intent, over ``library``'s tools."""

    def ex(utterance, tools, intent, *calls):
        return TaskExample(utterance, tuple(library[t] for t in tools), intent, calls)

    return [
        ex(
            "turn off the kitchen lights",
            ["set_lights", "set_thermostat", "play_music"],
            "call",
            Call("set_lights", {"room": "kitchen", "on": False}),
        ),
        ex(
            "dim the bedroom lights to 30% and play some jazz",
            ["play_music", "lock_door", "set_lights"],
            "call",
            Call("set_lights", {"room": "bedroom", "on": True, "brightness": 30}),
            Call("play_music", {"query": "jazz"}),
        ),
        ex(
            "turn on the lights",
            ["set_lights", "set_blinds"],
            "ask",
            Call("set_lights", {"room": "?", "on": True}),
        ),
        ex("can you change it", ["set_thermostat", "set_blinds"], "ask"),  # no slot labels
        ex("order me a pizza", ["set_lights", "play_music", "lock_door"], "unsupported"),
        ex("thanks, that's all", ["set_volume"], "none"),
    ]


def write_task(path: Path, library: dict[str, Tool]) -> None:
    """train.jsonl and the sets fine-tuning reads, all in ``path`` (its task and evals dir)."""
    path.mkdir()
    rows = [e.to_json() for e in examples(library)]
    for name in ("train", "val", "heldout-templates"):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


JOINT = {"task": "questions", "joint": True, "val_intent_acc": 0.9}


def save_joint(path: Path, tokenizer: BPETokenizer, meta: dict[str, object] = JOINT) -> Path:
    """A tiny Eve checkpoint with ``meta`` (a fine-tuned joint model by default)."""
    torch.manual_seed(0)
    save_checkpoint(path, Eve(preset("tiny")), tokenizer, dict(meta))
    return path


def save_retriever(path: Path, tokenizer: BPETokenizer) -> Path:
    """A tiny retriever checkpoint."""
    config = RetrieverConfig(**preset("tiny").to_dict() | {"embed_dim": 16})
    save_checkpoint(path, Retriever(config), tokenizer, {"task": "retrieval"})
    return path


# ---- training runs, as the golden and end-to-end tests start them ------------------------


def _pretrain_experiment(
    model: dict[str, Any],
    train_cfg: TrainConfig,
    data_dir: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    device: str | torch.device | None = None,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """The experiment ``pretrain`` runs, as config sections (see ``eve.experiment``):
    span-corruption data, loss and metrics, and fill-in samples. ``model`` is the start of
    the ``[model]`` table: a ``config`` or a ``preset`` and overrides."""
    return {
        "components": {
            "data": "span_corruption",
            "loss": "span",
            "metrics": "span",
            "report": "span_samples",
        },
        "model": {**model, "tokenizer": str(tokenizer_path)},
        "data": {"path": str(data_dir)},
        "train": train_cfg.to_dict(),
        "run": {
            "out": str(out_dir),
            "device": None if device is None else str(device),
            "resume": resume,
            "stop_after": stop_after,
        },
    }


def pretrain(
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_dir: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    device: str | torch.device,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, float]:
    """Span-corruption pretraining of the whole encoder-decoder on ``data_dir``'s
    ``train.bin``, evaluated on ``val.bin``. Each eval logs fill-in-the-blank samples; the
    last eval's are shown when the run ends.

    The input length is the model's ``max_encoder_len``, so every encoder position that
    fine-tuning uses has been trained (learned positions only learn where they are used).
    """
    values = _pretrain_experiment(
        {"config": cfg.to_dict()}, train_cfg, data_dir, tokenizer_path, out_dir, device,
        resume, stop_after,
    )  # fmt: skip
    return run(Experiment.from_dict(values))


def _finetune_experiment(
    train_cfg: TrainConfig,
    task_dir: str | Path,
    out_dir: str | Path,
    device: str | torch.device | None = None,
    init: str | Path | None = None,
    scratch: ModelConfig | str | None = None,
    tokenizer_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    evals_dir: str | Path = defaults.EVALS_DIR,
) -> dict[str, Any]:
    """The experiment ``finetune`` runs, as config sections (see ``eve.experiment``): task
    data, the joint loss and metrics, and the ``task_scores`` report. ``scratch`` is a config
    or a preset name."""
    if (init is None) == (scratch is None):
        raise ValueError("give either init (a checkpoint) or scratch (a config)")
    model: dict[str, Any] = dict(overrides or {})
    if init is not None:
        model["init"] = str(init)
    elif scratch is not None:
        start = {"preset": scratch} if isinstance(scratch, str) else {"config": scratch.to_dict()}
        tokenizer = None if tokenizer_path is None else str(tokenizer_path)
        model |= {**start, "tokenizer": tokenizer}  # None: refused, a random model has none
    return {
        "components": {
            "data": "task",
            "loss": "joint",
            "metrics": "joint",
            "report": "task_scores",
        },
        "model": model,
        "data": {"path": str(task_dir), "evals": str(evals_dir)},
        "train": train_cfg.to_dict(),
        "run": {
            "out": str(out_dir),
            "device": None if device is None else str(device),
            "resume": resume,
            "stop_after": stop_after,
        },
        "meta": {"task": TASK},  # joint and init: see trained_meta
    }


def finetune(
    train_cfg: TrainConfig,
    task_dir: str | Path,
    out_dir: str | Path,
    device: str | torch.device,
    library: dict[str, Tool],
    init: str | Path | None = None,
    scratch: ModelConfig | None = None,
    tokenizer_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    evals_dir: str | Path = defaults.EVALS_DIR,
) -> dict[str, float]:
    """Fine-tune the question heads and the decoder's argument values together
    (``joint_loss``) on ``task_dir``'s ``train.jsonl``, evaluated on ``evals_dir``'s
    ``val.jsonl``, with held-out-tool accuracy from its ``heldout-templates.jsonl`` at each eval.

    Start from ``init`` (a pretraining checkpoint or an earlier question run, see ``load_init``)
    or, to measure what pretraining buys, from a random ``scratch`` config
    and the tokenizer at ``tokenizer_path``.
    """
    values = _finetune_experiment(
        train_cfg, task_dir, out_dir, device, init, scratch, tokenizer_path, overrides, resume,
        stop_after, evals_dir,
    )  # fmt: skip
    return run(Experiment.from_dict(values), library=library)
