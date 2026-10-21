from pathlib import Path
from typing import Any, cast

import torch

from eve.config import ModelConfig
from eve.data.modules import task
from eve.data.schema import Tool
from eve.model.build import start_model
from eve.model.eve import Eve
from eve.runtime import seed_everything
from eve.tasks.questions import (
    TASK,
    joint_loss,
    joint_metrics,
    question_loss,
    question_metrics,
    task_scores,
)
from eve.train.config import TrainConfig
from eve.train.loop import train


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
    started = start_model(
        init=init,
        config=None if scratch is None else scratch.to_dict(),
        tokenizer=tokenizer_path,
        freeze=freeze,
        sdpa=sdpa,
        **(overrides or {}),
    )
    model, tokenizer = cast(Eve, started.model), started.tokenizer
    cfg = model.cfg
    data = task(tokenizer, cfg, train_cfg.seed, path=str(task_dir), library=library)
    joint = not freeze
    meta = {
        "task": TASK,
        "frozen": freeze,
        "joint": joint,  # the decoder was trained to write argument values too
        "trainable": started.trainable,
        "init": str(init or "scratch"),
    }
    return train(
        model,
        joint_loss if joint else question_loss,
        data,
        train_cfg,
        out_dir,
        tokenizer,
        device,
        report=task_scores(data, tokenizer, train_cfg.batch_size, values=joint),
        eval_fn=joint_metrics if joint else question_metrics,
        resume=resume,
        stop_after=stop_after,
        meta=meta,
    )
