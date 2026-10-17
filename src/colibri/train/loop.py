import json
import math
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from colibri import progress
from colibri.model.checkpoint import load_checkpoint, save_checkpoint
from colibri.model.colibri import Colibri
from colibri.tokenizer.bpe import BPETokenizer

LossFn = Callable[[Colibri, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
Collate = Callable[[Sequence[Any]], dict[str, torch.Tensor]]
DTYPES = {"float32": None, "bfloat16": torch.bfloat16}
RUN_FIELDS = ("eval_every", "eval_batches", "save_every", "log_every")  # may change on resume


@dataclass(frozen=True)
class TrainConfig:
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

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {self.dtype!r}")

    def to_dict(self) -> dict:
        return asdict(self)


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


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
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
    decay, no_decay = [], []
    for p in model.parameters():  # each tensor once, even the tied one
        (decay if id(p) in matrices else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def batch_indices(step: int, n_items: int, cfg: TrainConfig) -> np.ndarray:
    """The dataset indices of batch ``step``: each epoch is a new shuffle seeded by
    ``(seed, epoch)``, cut into whole batches (the last partial one is dropped). The order
    depends only on the step, so a resumed run sees exactly the batches it would have seen."""
    per_epoch = n_items // cfg.batch_size
    if per_epoch == 0:
        raise ValueError(f"{n_items} items is less than one batch of {cfg.batch_size}")
    epoch, i = divmod(step, per_epoch)
    order = np.random.default_rng((cfg.seed, epoch)).permutation(n_items)
    return order[i * cfg.batch_size : (i + 1) * cfg.batch_size]


def _rng_state() -> dict:
    state = {"torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _to(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: Colibri, loss_fn: LossFn, batches: Sequence[dict], device: torch.device, autocast
) -> dict[str, float]:
    """Mean of each loss over the val batches, with dropout off (book ch. 5.1.3)."""
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    for batch in batches:
        with autocast():
            losses = loss_fn(model, _to(batch, device))
        for name, loss in losses.items():
            totals[name] = totals.get(name, 0.0) + loss.float()
    model.train()
    return {f"val_{name}": (total / len(batches)).item() for name, total in totals.items()}


def train(
    model: Colibri,
    loss_fn: LossFn,
    train_data: Dataset,
    val_data: Dataset,
    collate: Collate,
    cfg: TrainConfig,
    out_dir: str | Path,
    tokenizer: BPETokenizer,
    device: str | torch.device,
    on_eval: Callable[[Colibri], dict] | None = None,
    eval_fn: LossFn | None = None,
    resume: bool = False,
    stop_after: int | None = None,
    meta: dict | None = None,
) -> dict[str, float]:
    """The training loop for pretraining and fine-tuning (book ch. 5.2 ``train_model_simple``
    plus App. D's warmup, cosine decay and gradient clipping).

    Each step: a batch -> ``loss_fn`` gives named losses -> their sum is backpropagated ->
    gradients are clipped to norm ``grad_clip`` -> AdamW step at ``lr_at(step)``.

    ``out_dir`` gets:
    - ``metrics.jsonl``: one JSON line per log (``kind`` "train"), eval ("val") and sample
      set ("samples", whatever ``on_eval`` returns);
    - ``last.pt``: model + optimizer + step + RNG states, written every ``save_every`` steps;
    - ``model.pt``: the final weights without the optimizer, for ``load_pretrained`` (box 7).

    Evals run ``eval_fn`` (default: ``loss_fn``) on the val batches. It may return extra
    metrics (e.g. an accuracy) that are logged with a ``val_`` prefix but never trained on.

    ``meta`` (plain values) is added to both checkpoints' meta, e.g. what kind of run made them.
    A dataset with a ``set_epoch`` method (``TaskDataset``) is told the epoch of each batch, so
    it can shuffle differently per epoch.

    ``resume=True`` continues from ``last.pt`` exactly: the same batches, dropout masks and
    optimizer state, so the losses match an uninterrupted run. ``stop_after`` ends the run
    early at that step (saving ``last.pt``), as if it had been interrupted.

    Returns the last val losses.
    """
    out_dir, device = Path(out_dir), torch.device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=device.type in ("cuda", "mps"),
    )
    dtype = DTYPES[cfg.dtype]
    autocast = (lambda: torch.autocast(device.type, dtype=dtype)) if dtype else nullcontext

    step = 0
    if resume and (out_dir / "last.pt").exists():
        step = _resume(model, optimizer, out_dir, cfg, device)
    elif resume:
        print(f"nothing to resume in {out_dir}: starting a new run")
    if step == 0:
        metrics_path.write_text("")

    def log(record: dict) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def save() -> None:
        state = {"step": step, "optimizer": optimizer.state_dict(), "rng": _rng_state()}
        info = {**(meta or {}), "step": step, "train_config": cfg.to_dict(), **last_val}
        save_checkpoint(out_dir / "last.pt", model, tokenizer, info, state)

    val_batches = [
        collate([val_data[j] for j in range(i * cfg.batch_size, (i + 1) * cfg.batch_size)])
        for i in range(min(cfg.eval_batches, len(val_data) // cfg.batch_size))
    ]
    if not val_batches:
        raise ValueError(f"val data has {len(val_data)} items, less than one batch")

    end = cfg.steps if stop_after is None else min(stop_after, cfg.steps)
    bar = progress.bar(end, out_dir.name, "steps", initial=step)  # to where this run stops

    def run_eval() -> None:
        before = dict(last_val)
        last_val.clear()
        last_val.update(evaluate(model, eval_fn or loss_fn, val_batches, device, autocast))
        log({"kind": "val", "step": step, **last_val})
        parts = []
        for k, v in last_val.items():
            change = ""
            if k in before:
                change = f" ({'+' if v >= before[k] else '-'}{_fmt(k, abs(v - before[k]))})"
            parts.append(f"{k} {progress.style(_fmt(k, v), '1')}{change}")
        bar.write(f"{progress.style('eval', '1', '35')} step {step:,} · " + " · ".join(parts))
        if on_eval is not None:
            samples = on_eval(model)
            log({"kind": "samples", "step": step, **samples})
            for line in samples.get("samples", []):
                bar.write(progress.style(f"    {line}", "2"))

    last_val: dict[str, float] = {}
    if step == 0:
        log({"kind": "start", "train_config": cfg.to_dict(), "model_config": model.cfg.to_dict()})
        run_eval()  # at step 0 the loss should be about ln(vocab_size): nothing learned yet

    window = {"loss": {}, "tokens": 0, "start": time.time(), "steps": 0}
    while step < end:
        idx = batch_indices(step, len(train_data), cfg)
        if hasattr(train_data, "set_epoch"):
            train_data.set_epoch(step // (len(train_data) // cfg.batch_size))
        batch = collate([train_data[int(j)] for j in idx])
        window["tokens"] += int(batch["enc_mask"].sum() + batch["dec_mask"].sum())

        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast():
            losses = loss_fn(model, _to(batch, device))
        loss = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        # Summed as tensors: calling .item() every step would make the CPU wait for the GPU.
        for name, value in losses.items():
            window["loss"][name] = window["loss"].get(name, 0.0) + value.detach().float()
        window["steps"] += 1
        bar.update()

        if step % cfg.log_every == 0 or step == end:
            elapsed = time.time() - window["start"]
            means = {k: (v / window["steps"]).item() for k, v in window["loss"].items()}
            tok_s = window["tokens"] / elapsed
            log(
                {
                    "kind": "train",
                    "step": step,
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                    "tok_s": tok_s,
                    "s_per_step": elapsed / window["steps"],
                    **means,
                }
            )
            stats = [f"{k} {v:.4f}" for k, v in means.items()]
            stats += [f"lr {lr:.1e}", f"{tok_s / 1e3:.1f}k tok/s"]
            if bar.disable:  # not a terminal: plain lines for the log file
                eta = (end - step) * elapsed / window["steps"]
                print(f"step {step:>6}/{end} | " + " | ".join(stats) + f" | eta {_hms(eta)}")
            bar.set_stats(*stats)
            window = {"loss": {}, "tokens": 0, "start": time.time(), "steps": 0}
        paused = time.time()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            run_eval()
        if step % cfg.save_every == 0 or step == end:
            save()
        window["start"] += time.time() - paused  # eval and saving don't count towards tok/s

    bar.close()
    if step == cfg.steps:
        info = {**(meta or {}), "step": step, **last_val}
        save_checkpoint(out_dir / "model.pt", model, tokenizer, info)
        print(f"{progress.style('done', '1', '32')} {out_dir / 'model.pt'}")
    else:
        print(f"{progress.style('stopped', '1', '33')} at step {step:,}; continue with --resume")
    return last_val


def _resume(
    model: Colibri,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> int:
    ckpt = load_checkpoint(out_dir / "last.pt")
    if ckpt.model.cfg != model.cfg:
        raise ValueError(f"{out_dir}/last.pt has a different model config")
    saved, given = ckpt.meta["train_config"], cfg.to_dict()
    if changed := [k for k in given if k not in RUN_FIELDS and saved.get(k) != given[k]]:
        raise ValueError(
            f"can't resume with a different {', '.join(changed)}: "
            + ", ".join(f"{k} was {saved.get(k)!r}" for k in changed)
        )
    model.load_state_dict(ckpt.model.state_dict())
    optimizer.load_state_dict(ckpt.train_state["optimizer"])  # moves its state to the device
    _set_rng_state(ckpt.train_state["rng"])
    step = ckpt.train_state["step"]

    # Drop metrics logged after the checkpoint: those steps run again now.
    path = out_dir / "metrics.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    kept = [ln for ln in lines if json.loads(ln).get("step", 0) <= step]
    path.write_text("".join(ln + "\n" for ln in kept))
    print(f"resumed {out_dir} at step {step}")
    return step


def _fmt(name: str, value: float) -> str:
    return f"{value:.1%}" if name.endswith(("acc", "rate")) else f"{value:.4f}"


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"
