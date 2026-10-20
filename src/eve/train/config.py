from dataclasses import asdict, dataclass
from typing import Any

import torch

DTYPES = {"float32": None, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer, schedule and run settings for ``train``."""

    steps: int
    batch_size: int = 32
    lr: float = 1e-3  # peak learning rate, reached at the end of warmup
    min_lr_ratio: float = 0.1  # cosine decay ends at lr * min_lr_ratio
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "float32"  # "bfloat16" runs the forward pass under autocast
    seed: int = 0  # batch order; the model's init is seeded by the caller
    eval_every: int = 500
    eval_batches: int = 40  # the first eval_batches * batch_size val items, always the same
    save_every: int = 500
    log_every: int = 20
    compile: bool = False  # torch.compile the training and eval forward passes (see train)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {self.dtype!r}")

    def to_dict(self) -> dict[str, Any]:
        """The config as plain values, for checkpoints and logs."""
        return asdict(self)
