import os
import random

import numpy as np
import torch


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
