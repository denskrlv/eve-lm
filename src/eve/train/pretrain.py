from pathlib import Path
from typing import cast

import torch

from eve.config import ModelConfig
from eve.data.modules import span_corruption
from eve.data.pretrain import INPUT_LEN, raw_length
from eve.model.build import start_model
from eve.model.eve import Eve
from eve.runtime import seed_everything
from eve.tasks.span import SAMPLES, span_loss, span_metrics, span_samples
from eve.train.config import TrainConfig
from eve.train.loop import train


def pretrain(
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_dir: str | Path,
    tokenizer_path: str | Path,
    out_dir: str | Path,
    device: str | torch.device,
    resume: bool = False,
    stop_after: int | None = None,
    sdpa: bool = True,
    samples: tuple[str, ...] = SAMPLES,
) -> dict[str, float]:
    """Box 4: span-corruption pretraining of the whole encoder-decoder on ``data_dir``'s
    ``train.bin``, evaluated on ``val.bin``. Each eval logs fill-in-the-blank samples; the
    last eval's are shown when the run ends.

    The input length is the model's ``max_encoder_len``, so every encoder position that
    fine-tuning uses has been trained (learned positions only learn where they are used).
    """
    data_dir = Path(data_dir)
    seed_everything(train_cfg.seed)
    started = start_model(config=cfg.to_dict(), tokenizer=tokenizer_path, sdpa=sdpa)
    model, tokenizer = cast(Eve, started.model), started.tokenizer
    data = span_corruption(tokenizer, cfg, train_cfg.seed, path=str(data_dir))

    return train(
        model,
        span_loss,
        data,
        train_cfg,
        out_dir,
        tokenizer,
        device,
        report=span_samples(data, tokenizer, train_cfg.batch_size, samples),
        eval_fn=span_metrics,
        resume=resume,
        stop_after=stop_after,
    )


def steps_for(tokens: int, batch_size: int, input_len: int = INPUT_LEN) -> int:
    """Optimizer steps to read ``tokens`` corpus tokens once."""
    return tokens // (batch_size * raw_length(input_len))
