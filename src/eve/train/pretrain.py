from pathlib import Path
from typing import Any

import torch

from eve.config import ModelConfig
from eve.data.pretrain import INPUT_LEN, raw_length
from eve.experiment import Experiment, run
from eve.tasks.span import SAMPLES
from eve.train.config import TrainConfig


def pretrain_experiment(
    model: dict[str, Any],
    train_cfg: TrainConfig,
    data_dir: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    device: str | torch.device | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    sdpa: bool = True,
    samples: tuple[str, ...] = SAMPLES,
) -> dict[str, Any]:
    """The experiment ``pretrain`` runs, as config sections (see ``eve.experiment``):
    span-corruption data, loss and metrics, and fill-in samples. ``model`` is the start of
    the ``[model]`` table: a ``config`` or a ``preset`` and overrides."""
    report = {} if tuple(samples) == SAMPLES else {"samples": list(samples)}
    return {
        "components": {
            "data": "span_corruption",
            "loss": "span",
            "metrics": "span",
            "report": "span_samples",
        },
        "model": {**model, "tokenizer": str(tokenizer_path), "sdpa": sdpa},
        "data": {"path": str(data_dir)},
        "report": report,
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
    sdpa: bool = True,
    samples: tuple[str, ...] = SAMPLES,
) -> dict[str, float]:
    """Box 4: span-corruption pretraining of the whole encoder-decoder on ``data_dir``'s
    ``train.bin``, evaluated on ``val.bin``. Each eval logs fill-in-the-blank samples; the
    last eval's are shown when the run ends.

    The input length is the model's ``max_encoder_len``, so every encoder position that
    fine-tuning uses has been trained (learned positions only learn where they are used).
    """
    values = pretrain_experiment(
        {"config": cfg.to_dict()}, train_cfg, data_dir, tokenizer_path, out_dir, device,
        resume, stop_after, sdpa, samples,
    )  # fmt: skip
    return run(Experiment.from_dict(values))


def steps_for(tokens: int, batch_size: int, input_len: int = INPUT_LEN) -> int:
    """Optimizer steps to read ``tokens`` corpus tokens once."""
    return tokens // (batch_size * raw_length(input_len))
