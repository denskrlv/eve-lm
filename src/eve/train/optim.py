"""The registered optimizers and learning-rate schedules (kinds "optimizer" and "schedule")."""

import math
from typing import Any

import torch
from torch import nn

from eve.registry import register
from eve.train.config import TrainConfig

ENCODER = ("enc_embeddings.", "encoder.")  # a MiniLM encoder's weights (Eve.enc_case is new)


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


def param_groups(
    model: nn.Module, weight_decay: float, encoder_lr_scale: float = 1.0
) -> list[dict[str, Any]]:
    """Weight decay only on the weight matrices of linear layers.

    Decay pulls every weight a little towards 0 each step. That makes sense for matrices,
    where it keeps any one connection from growing huge. It doesn't for LayerNorm's scale
    (pulling it to 0 would switch features off), for biases, or for embeddings: a rare token's
    row gets few gradient updates, so decay would shrink it towards 0 between them. The tied LM
    head is the token embedding, so it is not decayed either.

    With ``encoder_lr_scale`` other than 1, a pretrained encoder's weights (``ENCODER``: MiniLM's
    in an Eve) get groups of their own with ``lr_scale``, which the loop multiplies the
    schedule's lr by: small steps keep what MiniLM learned while the new heads learn fast.
    """
    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    matrices = {
        id(m.weight)
        for m in model.modules()
        if isinstance(m, nn.Linear) and id(m.weight) not in embeddings
    }
    encoder = (
        {id(p) for name, p in model.named_parameters() if name.startswith(ENCODER)}
        if encoder_lr_scale != 1.0
        else set()
    )
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    for param in model.parameters():  # each tensor once, even the tied one
        groups[id(param) in encoder, id(param) in matrices].append(param)
    out = []
    for (in_encoder, decayed), params in groups.items():
        if in_encoder and not params:
            continue
        group: dict[str, Any] = {"params": params, "weight_decay": weight_decay if decayed else 0.0}
        if in_encoder:
            group["lr_scale"] = encoder_lr_scale
        out.append(group)
    return out


@register("optimizer", "adamw")
def adamw(model: nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW with ``cfg``'s lr, betas and weight decay (``param_groups``); the fused kernel on
    CUDA and MPS."""
    return torch.optim.AdamW(
        param_groups(model, cfg.weight_decay, cfg.encoder_lr_scale),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=device.type in ("cuda", "mps"),
    )
