# eve-lm

A ~32M-parameter on-device model that turns requests into tool calls with calibrated
confidences. Python 3.12+, PyTorch, managed with uv. Package code is in `src/eve/`, the CLI entry
point is `eve` (`src/eve/cli.py`).

## Checks

Run what CI runs before calling a change done:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy        # strict, src only
uv run pytest -q
```

Golden tests (`tests/golden/`) pin training numbers. Rewrite them only for an intended change in
behavior: `EVE_GOLDEN=write uv run pytest tests/test_golden.py`, and say why in the commit.

## Model and experiments

- One model: `eve-32m`. `ModelConfig` defaults are its sizes; the `tiny` preset exists for tests.
- Experiments live in `configs/experiments/` and run with `uv run eve run <file>`; params layer
  as component file → experiment table → `--set section.key=value`.
- Runs write to `artifacts/runs/<name>/`. `artifacts/`, `data/bin*`, `data/task/` and
  `data/evals/` are generated and gitignored.
- Pretraining runs on Kaggle (`notebooks/kaggle-train.ipynb`); fine-tuning and evaluation run
  locally.

## Local training on Apple Silicon

MPS shares system memory and its allocator grows with every new tensor shape. Before a local run:
estimate peak memory, pad batches to fixed shapes, smoke-test a few steps, and cap memory with
both `PYTORCH_MPS_HIGH_WATERMARK_RATIO` and `PYTORCH_MPS_LOW_WATERMARK_RATIO` (e.g. 0.3 / 0.2).
The joint fine-tune (`eve-32m.toml`) OOMs in eager fp32: launch it with
`--set train.compile=true --set train.dtype=bfloat16` and ratios of about 0.45 / 0.35.

## Working style

- Small changes (a fix, a rename, a doc edit): skip brainstorming and written plans.
- Use TDD for library, data and tokenizer code, not for training experiments; judge those by
  eval numbers (`uv run eve evaluate artifacts/runs/<name>`).
- Local planning notes live in `md/` (gitignored); the roadmap is `md/PLAN.md` when present.
