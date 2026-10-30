"""Publish a fine-tuned model (M10): ``eve export`` writes a directory for a Hugging Face Hub
model repo, a ``model.pt`` that ``eve.Eve.load`` reads and a ``README.md`` model card, and with
``retriever``, a ``retriever.pt`` that ``Eve.load`` finds beside it (M11).

The exported checkpoint keeps the architecture, config, tokenizer and meta (calibration
included) but no training state, and stores the weights in ``dtype``: float16 halves the
download, and loading casts them back to float32.
"""

from pathlib import Path
from typing import Any

import torch

from eve.model.checkpoint import (
    MODEL_FILE,
    RETRIEVAL_TASK,
    RETRIEVER_FILE,
    load_checkpoint,
    save_checkpoint,
)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
METRICS = {  # the validation scores shown in the model card, by meta key
    "val_intent_acc": "intent accuracy",
    "val_exact_acc": "exact (intent and tools)",
    "val_value_acc": "argument values (given the right tools)",
    "val_full_acc": "whole call right",
}


def export_model(
    model: str | Path,
    out: str | Path,
    dtype: str = "float16",
    repo: str = "<user>/<repo>",
    retriever: str | Path | None = None,
) -> Path:
    """Write ``out/model.pt`` and ``out/README.md`` from the joint checkpoint ``model``, and
    ``out/retriever.pt`` from the ``retriever`` checkpoint if given."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    checkpoint = load_checkpoint(model)
    if not checkpoint.meta.get("joint"):
        raise ValueError(f"{model} isn't a fine-tuned joint model; only those can be published")
    out = Path(out)
    meta = checkpoint.meta | {"dtype": dtype, "exported_from": str(model)}
    save_checkpoint(
        out / MODEL_FILE, checkpoint.model.to(DTYPES[dtype]), checkpoint.tokenizer, meta
    )
    params = sum(p.numel() for p in checkpoint.model.parameters())
    if retriever is not None:
        found = load_checkpoint(retriever)
        if found.meta.get("task") != RETRIEVAL_TASK:
            raise ValueError(f"{retriever} isn't a retriever")
        extra = found.meta | {"dtype": dtype, "exported_from": str(retriever)}
        weights = found.model.to(DTYPES[dtype])
        save_checkpoint(out / RETRIEVER_FILE, weights, found.tokenizer, extra)
    (out / "README.md").write_text(model_card(meta, params, repo, retriever is not None))
    return out


def model_card(meta: dict[str, Any], params: int, repo: str, retriever: bool = False) -> str:
    """A Hub model card: what the model does, how to use it, its scores and its limits."""
    scores = "\n".join(
        f"| {label} | {meta[key]:.1%} |" for key, label in METRICS.items() if key in meta
    )
    calibrated = "calibration" in meta
    calibration = (
        "Probabilities are temperature-scaled: one set of temperatures for tools seen in "
        "training, another for new tools, picked per request."
        if calibrated
        else "Probabilities are as trained (not calibrated)."
    )
    picking = (
        "with more registered (`eve.add_tools`), the retriever in `retriever.pt` picks the 5 "
        "a request reads, by meaning"
        if retriever
        else "with more registered (`eve.add_tools`), BM25 word overlap picks the 5 a request reads"
    )
    return f"""---
license: apache-2.0
library_name: eve-lm
tags:
  - function-calling
  - tool-calling
  - on-device
---

# Eve

Eve turns a request and a few tool schemas into a decision: call tools (with argument values),
ask the user for a missing argument, or say no tool can do it, each with a probability. It has
{params / 1e6:.1f}M parameters and runs on a CPU.

```bash
pip install eve-lm
```

```python
from eve import Eve, tool

@tool
def set_lights(room: str, on: bool):
    \"\"\"Turn the lights in a room on or off.\"\"\"

eve = Eve.load("{repo}")
response = eve.respond("turn off the kitchen lights", tools=[set_lights])
print(response.intent, [(c.tool, c.args) for c in response.calls])
```

## Validation scores

| measure | score |
|---|---|
{scores}

{calibration}

## Limits

- Up to 5 tools per request; {picking}. Arguments are str, int, float, bool or an enum of
  strings.
- Tools like the ones in training work best. On new tools, especially when the request shares
  no word with the tool's name, accuracy is much lower, and the probabilities say so less
  reliably.
- English only.

Code, training and evaluation: https://github.com/denskrlv/eve-lm
"""
