import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from eve import registry
from eve.config import ModelConfig
from eve.model.eve import Eve
from eve.tokenizer.bpe import BPETokenizer

FORMAT = "eve-ckpt-1"
FRESH = "questions."  # the question block: pretraining never trains it, fine-tuning starts it new
QUESTION_FIELDS = frozenset(  # config fields that change no pretrained weight's shape
    {"n_question_layers", "n_slots", "max_tools", "pointer_dim", "dropout"}
)


@dataclass
class Checkpoint:
    """A loaded checkpoint: model, tokenizer, meta and, for resuming, train state."""

    model: Eve
    tokenizer: BPETokenizer
    meta: dict[str, Any]
    train_state: dict[str, Any] | None = None  # optimizer, step and RNG states, for resuming (M4)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    tokenizer: BPETokenizer,
    meta: dict[str, Any] | None = None,
    train_state: dict[str, Any] | None = None,
) -> None:
    """Save everything needed to rebuild the model: ``{format, model, config, state_dict,
    tokenizer, meta}``, where ``model`` is the architecture's registered name. The book
    (ch. 5.4) saves only ``model.state_dict()`` and rebuilds the model from a config it keeps
    in code; storing the architecture, config and tokenizer too means a checkpoint file can't
    be paired with the wrong ones, and loads without knowing the code path that made it.

    ``meta`` is for things like the step, tokens seen and val loss. Keep it to plain Python
    values (numbers, strings, lists, dicts): files are loaded with ``weights_only=True``,
    which refuses arbitrary pickled objects. ``train_state`` (the training loop's optimizer,
    step and RNG states) is only needed to resume a run; the final model is saved without it.

    The file is written to a temporary name and then renamed, so a crash during a save never
    leaves a half-written checkpoint behind.
    """
    cfg = getattr(model, "cfg")  # noqa: B009 (nn.Module's __getattr__ hides it from mypy)
    if tokenizer.vocab_size > cfg.vocab_size:
        raise ValueError(
            f"tokenizer has {tokenizer.vocab_size} tokens, model only {cfg.vocab_size}"
        )
    data = {
        "format": FORMAT,
        "model": _architecture(model),
        "config": cfg.to_dict(),
        "state_dict": model.state_dict(),  # the tied LM head is saved as a shared tensor
        "tokenizer": tokenizer.to_dict(),
        "meta": meta or {},
    }
    if train_state is not None:
        data["train_state"] = train_state
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


def _architecture(model: nn.Module) -> str:
    """The registered name of ``model``'s class, or of the nearest registered base class (a
    test's subclass of ``Eve`` saves as "eve")."""
    for cls in type(model).__mro__:
        try:
            return registry.name_of("model", cls)
        except KeyError:
            continue
    raise KeyError(f"{type(model).__name__} is not a registered model")


def _read(path: str | Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path} is not an Eve checkpoint ({FORMAT})")
    return data


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Checkpoint:
    """Rebuild the exact model that was saved (strict: every key must match). For resuming a
    run and for inference. The model comes back in train mode, like a new ``Eve``."""
    data = _read(path)
    architecture = registry.get("model", data.get("model", "eve"))  # older files: all Eve
    model: Eve = architecture.from_config(data["config"])  # typed Eve: what every reader uses
    model.load_state_dict(data["state_dict"])
    tokenizer = BPETokenizer.from_dict(data["tokenizer"])
    return Checkpoint(model.to(device), tokenizer, data["meta"], data.get("train_state"))


def load_pretrained(
    path: str | Path, device: str | torch.device = "cpu", **overrides: Any
) -> Checkpoint:
    """Box 7: start fine-tuning from the M4 pretraining checkpoint.

    The book (ch. 5.5) copies OpenAI's GPT-2 weights into a fresh ``GPTModel``, and ch. 6 then
    swaps the output layer for a new, random classifier head. Here the encoder, decoder and
    embeddings come from span-corruption pretraining, and the question block (``questions.*``)
    starts random: pretraining has no labels for it, so its saved weights are only the
    initialization. Seed before calling to make that initialization repeatable.

    ``overrides`` may change the question block's config (e.g. ``n_slots=8``, ``dropout=0.0``)
    because its weights aren't loaded anyway; changing the body (``d_model``, layer counts…)
    would leave the pretrained weights without a home, so it is refused.

    Loading is non-strict, then checked: the only missing keys may be ``questions.*`` and no
    saved key may be left over. A silently skipped encoder weight would otherwise train from
    random and look like "pretraining didn't help".
    """
    if bad := set(overrides) - QUESTION_FIELDS:
        raise ValueError(f"can't change {sorted(bad)} of a pretrained body")
    data = _read(path)
    model = Eve(replace(ModelConfig(**data["config"]), **overrides))
    body = {k: v for k, v in data["state_dict"].items() if not k.startswith(FRESH)}
    result = model.load_state_dict(body, strict=False)
    if not_loaded := [k for k in result.missing_keys if not k.startswith(FRESH)]:
        raise KeyError(f"{path} has no weights for {not_loaded}")
    if result.unexpected_keys:
        raise KeyError(f"{path} has weights the model doesn't use: {result.unexpected_keys}")
    return Checkpoint(model.to(device), BPETokenizer.from_dict(data["tokenizer"]), data["meta"])
