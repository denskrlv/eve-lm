from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from eve.config import ModelConfig
from eve.data.jsonl import read_jsonl
from eve.data.render import TaskCollator, TaskDataset, TaskExample
from eve.data.schema import Tool
from eve.model.checkpoint import (
    FRESH,
    QUESTION_FIELDS,
    Checkpoint,
    load_checkpoint,
    load_pretrained,
)
from eve.model.eve import Eve
from eve.runtime import seed_everything
from eve.tasks.questions import (
    COLUMNS,
    HIGHLIGHTS,
    TASK,
    VAL_SEED,
    eval_report,
    joint_loss,
    joint_metrics,
    question_loss,
    question_metrics,
    summary,
)
from eve.tokenizer.bpe import BPETokenizer
from eve.train.config import TrainConfig
from eve.train.loop import Report, train


def freeze_body(model: Eve) -> int:
    """Stage 1 of ch. 6.5: train only the new head and keep the pretrained weights fixed.

    Here the "head" is the whole question block (``questions.*``): its learned queries, its
    two blocks and both heads. The embeddings, encoder and decoder get ``requires_grad=False``,
    so backprop stops at the encoder output and AdamW leaves them alone. Returns the number of
    parameters that still train.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(FRESH)
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def load_init(path: str | Path, device: str | torch.device = "cpu", **overrides: Any) -> Checkpoint:
    """The model to fine-tune, from either kind of checkpoint:

    - a pretraining checkpoint (box 7): ``load_pretrained``, so the question block starts new;
    - a checkpoint of an earlier question run (``meta["task"] == "questions"``), e.g. stage 1
      with the body frozen: everything is loaded, the trained question block too.

    ``overrides`` may change the question block's config, as in ``load_pretrained``; for an
    earlier question run only ``dropout`` (other changes would leave weights without a home).
    """
    if bad := set(overrides) - QUESTION_FIELDS:
        raise ValueError(f"can't change {sorted(bad)} of a pretrained body")
    ckpt = load_checkpoint(path)
    if ckpt.meta.get("task") != TASK:
        return load_pretrained(path, device, **overrides)
    model = Eve(replace(ckpt.model.cfg, **overrides))
    model.load_state_dict(ckpt.model.state_dict())
    ckpt.model = model.to(device)
    return ckpt


def load_examples(path: str | Path, library: dict[str, Tool]) -> list[TaskExample]:
    """The examples of a task .jsonl file, with tools looked up in ``library``."""
    return [TaskExample.from_json(row, library) for row in read_jsonl(path)]


def _start_model(
    init: str | Path | None,
    scratch: ModelConfig | None,
    tokenizer_path: str | Path | None,
    overrides: dict[str, Any],
) -> tuple[Eve, BPETokenizer]:
    """The model to fine-tune and its tokenizer: from the ``init`` checkpoint, or a random model
    with the ``scratch`` config and the tokenizer at ``tokenizer_path``."""
    if init is not None:
        ckpt = load_init(init, **overrides)
        return ckpt.model, ckpt.tokenizer
    assert scratch is not None  # finetune checks: exactly one of init and scratch
    if tokenizer_path is None:
        raise ValueError("scratch needs tokenizer_path")
    model = Eve(replace(scratch, **overrides))
    return model, BPETokenizer.load(tokenizer_path)


def _task_data(
    task_dir: Path, library: dict[str, Tool], tokenizer: BPETokenizer, cfg: ModelConfig
) -> tuple[TaskDataset, list[TaskExample], TaskDataset, TaskDataset]:
    """The train set, the val examples and set, and the held-out-tool set of ``task_dir``."""
    train_data = TaskDataset(load_examples(task_dir / "train.jsonl", library), tokenizer, cfg)
    val_examples = load_examples(task_dir / "val.jsonl", library)
    val_data = TaskDataset(val_examples, tokenizer, cfg, seed=VAL_SEED)
    heldout = TaskDataset(
        load_examples(task_dir / "heldout-templates.jsonl", library), tokenizer, cfg
    )
    return train_data, val_examples, val_data, heldout


def finetune(
    train_cfg: TrainConfig,
    task_dir: str | Path,
    out_dir: str | Path,
    device: str | torch.device,
    library: dict[str, Tool],
    init: str | Path | None = None,
    scratch: ModelConfig | None = None,
    tokenizer_path: str | Path | None = None,
    freeze: bool = False,
    overrides: dict[str, Any] | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    sdpa: bool = True,
) -> dict[str, float]:
    """Boxes 8 and 9: fine-tune the question heads and the decoder's argument values together
    (``joint_loss``) on ``task_dir``'s ``train.jsonl``, evaluated on ``val.jsonl``, with
    held-out-tool accuracy from ``heldout-templates.jsonl`` at each eval.

    Start from ``init`` (a pretraining checkpoint or an earlier question run, see ``load_init``)
    or, for the "what does pretraining buy?" experiment (E1), from a random ``scratch`` config
    and the tokenizer at ``tokenizer_path``. ``freeze`` trains only the question block (E2), so
    it trains on ``question_loss`` alone: the frozen decoder would learn nothing from its loss.
    """
    if (init is None) == (scratch is None):
        raise ValueError("give either init (a checkpoint) or scratch (a config)")
    task_dir = Path(task_dir)
    seed_everything(train_cfg.seed)  # the new question block's initialization
    model, tokenizer = _start_model(init, scratch, tokenizer_path, overrides or {})
    trainable = freeze_body(model) if freeze else sum(p.numel() for p in model.parameters())
    model.set_sdpa(sdpa)

    cfg = model.cfg
    train_data, val_examples, val_data, heldout = _task_data(task_dir, library, tokenizer, cfg)
    collate = TaskCollator(cfg)
    joint = not freeze
    on_eval = eval_report(val_examples, val_data, heldout, collate, train_cfg.batch_size, joint)
    meta = {
        "task": TASK,
        "frozen": freeze,
        "joint": joint,  # the decoder was trained to write argument values too
        "trainable": trainable,
        "init": str(init or "scratch"),
    }
    return train(
        model,
        joint_loss if joint else question_loss,
        train_data,
        val_data,
        collate,
        train_cfg,
        out_dir,
        tokenizer,
        device,
        report=Report(COLUMNS, on_eval, summary, HIGHLIGHTS),
        eval_fn=joint_metrics if joint else question_metrics,
        resume=resume,
        stop_after=stop_after,
        meta=meta,
    )
