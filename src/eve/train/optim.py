"""The registered optimizers and learning-rate schedules (kinds "optimizer" and "schedule")."""

import math
from typing import Any

import torch
from torch import nn

from eve.registry import register
from eve.train.config import TrainConfig


@register("schedule", "warmup_cosine")
def lr_at(step: int, cfg: TrainConfig) -> float:
    """Book App. D: linear warmup to ``lr``, then cosine decay to ``lr * min_lr_ratio``.

    Warmup: early gradients come from random weights and are large and noisy, and Adam's
    running averages are still empty, so big steps would throw the weights around. Cosine:
    large steps while the loss falls fast, small careful ones at the end.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    min_lr = cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return min_lr + (cfg.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay only on the weight matrices of linear layers.

    Decay pulls every weight a little towards 0 each step. That makes sense for matrices,
    where it keeps any one connection from growing huge. It doesn't for LayerNorm's scale
    (pulling it to 0 would switch features off), for biases, or for embeddings: a rare token's
    row gets few gradient updates, so decay would shrink it towards 0 between them. The tied LM
    head is the token embedding, so it is not decayed either.
    """
    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    matrices = {
        id(m.weight)
        for m in model.modules()
        if isinstance(m, nn.Linear) and id(m.weight) not in embeddings
    }
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():  # each tensor once, even the tied one
        (decay if id(param) in matrices else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@register("optimizer", "adamw")
def adamw(model: nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW with ``cfg``'s lr, betas and weight decay (``param_groups``); the fused kernel on
    CUDA and MPS."""
    return torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=device.type in ("cuda", "mps"),
    )
