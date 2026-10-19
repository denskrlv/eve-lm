import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


def pick_device(name: str | None = None) -> torch.device:
    """Use ``name`` if given (e.g. "cpu"), else the fastest available: CUDA, then MPS, then CPU."""
    name = name or os.environ.get("COLIBRI_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA and MPS) so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def eval_mode(model: nn.Module) -> Iterator[torch.device]:
    """Dropout off inside the block, the model's own mode back after it. Yields its device."""
    was_training = model.training
    model.eval()
    try:
        yield next(model.parameters()).device
    finally:
        model.train(was_training)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Every tensor of a batch on ``device``."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
