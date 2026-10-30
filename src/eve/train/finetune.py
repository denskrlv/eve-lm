from pathlib import Path
from typing import Any

import torch

from eve.config import ModelConfig
from eve.data.schema import Tool
from eve.experiment import Experiment, run
from eve.tasks.questions import TASK
from eve.train.config import TrainConfig


def finetune_experiment(
    train_cfg: TrainConfig,
    task_dir: str | Path,
    out_dir: str | Path,
    device: str | torch.device | None = None,
    init: str | Path | None = None,
    scratch: ModelConfig | str | None = None,
    tokenizer_path: str | Path | None = None,
    freeze: bool = False,
    overrides: dict[str, Any] | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    sdpa: bool = True,
) -> dict[str, Any]:
    """The experiment ``finetune`` runs, as config sections (see ``eve.experiment``): task
    data, the joint loss and metrics, or the question ones with ``freeze``, and the
    ``task_scores`` report. ``scratch`` is a config or a preset name."""
    if (init is None) == (scratch is None):
        raise ValueError("give either init (a checkpoint) or scratch (a config)")
    joint = not freeze
    model: dict[str, Any] = {"freeze": freeze, "sdpa": sdpa, **(overrides or {})}
    if init is not None:
        model["init"] = str(init)
    elif scratch is not None:
        start = {"preset": scratch} if isinstance(scratch, str) else {"config": scratch.to_dict()}
        tokenizer = None if tokenizer_path is None else str(tokenizer_path)
        model |= {**start, "tokenizer": tokenizer}  # None: refused, a random model has none
    name = "joint" if joint else "question"
    return {
        "components": {"data": "task", "loss": name, "metrics": name, "report": "task_scores"},
        "model": model,
        "data": {"path": str(task_dir)},
        "report": {"values": joint},
        "train": train_cfg.to_dict(),
        "run": {
            "out": str(out_dir),
            "device": None if device is None else str(device),
            "resume": resume,
            "stop_after": stop_after,
        },
        "meta": {"task": TASK, "frozen": freeze},  # joint and init: see trained_meta
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
    values = finetune_experiment(
        train_cfg, task_dir, out_dir, device, init, scratch, tokenizer_path, freeze,
        overrides, resume, stop_after, sdpa,
    )  # fmt: skip
    return run(Experiment.from_dict(values), library=library)
