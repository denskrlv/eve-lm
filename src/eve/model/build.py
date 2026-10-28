"""Starting a run's model (the ``[model]`` config table): random from a config, or from a
checkpoint, with the architecture registered under ``architecture`` (kind "model")."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eve import registry
from eve.config import preset as preset_config
from eve.model.checkpoint import (
    FRESH,
    QUESTION_FIELDS,
    Checkpoint,
    load_checkpoint,
    load_pretrained,
)
from eve.model.eve import Eve
from eve.tokenizer.bpe import BPETokenizer

QUESTIONS_TASK = "questions"  # meta["task"] of a question run's checkpoints (tasks.questions)


@dataclass
class Started:
    """A model ready to train, its tokenizer, and how many of its parameters train."""

    model: nn.Module
    tokenizer: BPETokenizer
    trainable: int


def freeze_body(model: nn.Module) -> int:
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
    if ckpt.meta.get("task") != QUESTIONS_TASK:
        return load_pretrained(path, device, **overrides)
    model = Eve(replace(ckpt.model.cfg, **overrides))
    model.load_state_dict(ckpt.model.state_dict())
    ckpt.model = model.to(device)
    return ckpt


def start_model(
    architecture: str = "eve",
    init: str | Path | None = None,
    preset: str | None = None,
    config: dict[str, Any] | None = None,
    tokenizer: str | Path | None = None,
    freeze: bool = False,
    sdpa: bool = True,
    **overrides: Any,
) -> Started:
    """The model a run starts from: the ``init`` checkpoint (``load_init``), or random weights
    with ``config``, or with a ``preset`` of ``eve.config.PRESETS``, and the tokenizer at
    ``tokenizer``. ``overrides`` change config fields (``dropout=0.0``). Seed before calling:
    random weights come from the global RNG.

    An architecture with a ``start`` classmethod (the ``Retriever``) starts from ``init`` its
    own way. ``freeze`` trains only the question block (``freeze_body``); ``sdpa`` picks the
    attention kernel (``Eve.set_sdpa``).
    """
    start = getattr(registry.get("model", architecture), "start", None)
    if init is not None and start is not None:  # an architecture that starts its own way
        ckpt = start(init, **overrides)
        model: nn.Module = ckpt.model
        loaded = ckpt.tokenizer
    elif init is not None:
        ckpt = load_init(init, **overrides)
        model = ckpt.model
        loaded = ckpt.tokenizer
    else:
        if config is None:
            if preset is None:
                raise ValueError("give init (a checkpoint), preset or config")
            config = preset_config(preset).to_dict()
        if tokenizer is None:
            raise ValueError("scratch needs tokenizer_path")  # a random model has no tokenizer
        model = registry.get("model", architecture).from_config(config | overrides)
        loaded = BPETokenizer.load(tokenizer)
    trainable = freeze_body(model) if freeze else sum(p.numel() for p in model.parameters())
    if (set_sdpa := getattr(model, "set_sdpa", None)) is not None:  # an Eve or a Retriever
        set_sdpa(sdpa)
    return Started(model, loaded, trainable)
