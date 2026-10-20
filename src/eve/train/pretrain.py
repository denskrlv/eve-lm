from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from eve import defaults
from eve.config import ModelConfig
from eve.data.corpus import load_tokens
from eve.data.pretrain import INPUT_LEN, SpanCorruptionDataset, raw_length, spans_by_sentinel
from eve.data.render import IGNORE, collate_pretrain
from eve.model.eve import Eve
from eve.runtime import eval_mode, seed_everything
from eve.tokenizer.bpe import BYTE0, DEC, EOS, BPETokenizer, is_sentinel
from eve.train.config import TrainConfig
from eve.train.loop import Column, Report, train

SAMPLES = (
    "The capital of France is <x0>.",
    "Water boils at 100 degrees <x0> at sea level.",
    "Plants use energy from <x0> to turn water and carbon dioxide into sugar.",
    "The Second World War ended in <x0>.",
    "Students who <x0> their homework every day usually <x1> better grades.",
    "Turn off the <x0> in the kitchen when you leave the room.",
)
COLUMNS = (
    Column("val loss", "val_lm"),
    Column("content loss", "val_content"),
    Column("content acc", "val_content_acc"),
)  # the eval table; see span_metrics
HIGHLIGHTS = ((r"\[[^\]]*\]", "bold cyan"),)  # fill_in's [filled-in spans]


def span_loss(model: Eve, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Box 4: cross-entropy of the decoder's next-token predictions against the missing spans
    (book ch. 5.1.2, ``calc_loss_batch``). Padding positions are labelled IGNORE."""
    out = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"])
    logits = out.lm.flatten(0, 1).float()  # (B * n_dec, vocab)
    return {"lm": F.cross_entropy(logits, batch["dec_labels"].flatten(), ignore_index=IGNORE)}


def prompt_ids(tokenizer: BPETokenizer, text: str) -> list[int]:
    """``text`` with ``<x0>``… holes -> encoder ids shaped like a pretraining input.

    In training a hole replaces whole tokens, and a word token carries its leading space
    (`` Paris``), so a sentinel never follows a lone space. Typed text has one
    (``"is <x0>."`` -> `` is``, `` ``, ``<x0>``), so that space token is dropped.
    """
    ids = tokenizer.encode(text, specials=True)
    return [
        t
        for i, t in enumerate(ids)
        if not (i + 1 < len(ids) and is_sentinel(ids[i + 1]) and tokenizer.decode([t]).isspace())
    ]


def span_metrics(model: Eve, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Box 6, the val metrics: the training loss ``lm`` plus the loss and top-1 accuracy on
    *content* tokens only, the words inside the holes.

    About a quarter of the labels are sentinels and ``<eos>``: they only mark where spans
    start and end, and the model gets them right early (~90%). They pull ``lm`` down without
    saying whether the model can fill a hole, so ``content`` is the number to watch.
    """
    out = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"])
    logits, labels = out.lm.float(), batch["dec_labels"]
    nll = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE, reduction="none"
    ).view_as(labels)
    real = labels != IGNORE
    content = labels >= BYTE0  # every special token (sentinels, <eos>, IGNORE) is below BYTE0
    return {
        "lm": nll[real].mean(),
        "content": nll[content].mean(),
        "content_acc": (logits.argmax(-1)[content] == labels[content]).float().mean(),
    }


@torch.no_grad()
def fill_in(
    model: Eve,
    tokenizer: BPETokenizer,
    text: str,
    max_new: int = defaults.MAX_NEW_TOKENS,
) -> str:
    """Fill the ``<x0>``, ``<x1>``… holes of ``text`` by greedy decoding (book ch. 5.1.1's
    ``generate_text_simple``, but the decoder also reads the encoder).

    The decoder writes ``<x0> span <x1> span … <eos>``, the same format as the pretraining
    targets. Returns ``text`` with each hole replaced by ``[its span]``, or ``[?]`` if the
    model didn't fill it.
    """
    with eval_mode(model) as device:
        enc = torch.tensor([prompt_ids(tokenizer, text)], device=device)
        memory = model.encode(enc, torch.ones_like(enc, dtype=torch.bool))
        dec = [DEC]
        for _ in range(max_new):
            ids = torch.tensor([dec], device=device)
            logits = model.decode(ids, memory, torch.ones_like(enc, dtype=torch.bool))
            next_id = int(logits[0, -1].argmax())
            if next_id == EOS:
                break
            dec.append(next_id)

    spans = spans_by_sentinel(dec[1:])
    out = []
    for token in enc[0].tolist():
        if token in spans:
            span = tokenizer.decode(spans[token])
            space = " " if span.startswith(" ") else ""  # " Paris" -> " [Paris]"
            out.append(f"{space}[{span.removeprefix(space)}]")
        elif is_sentinel(token):
            out.append(" [?]")  # a hole the model didn't fill
        else:
            out.append(tokenizer.decode([token]))
    return "".join(out)


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
    tokenizer = BPETokenizer.load(tokenizer_path)
    seed_everything(train_cfg.seed)
    model = Eve(cfg)
    model.set_sdpa(sdpa)
    train_data = SpanCorruptionDataset(
        load_tokens(data_dir / "train.bin"), cfg.max_encoder_len, seed=train_cfg.seed
    )
    val_data = SpanCorruptionDataset(load_tokens(data_dir / "val.bin"), cfg.max_encoder_len)

    def on_eval(model: Eve) -> dict[str, Any]:
        return {"samples": [fill_in(model, tokenizer, text) for text in samples]}

    return train(
        model,
        span_loss,
        train_data,
        val_data,
        collate_pretrain,
        train_cfg,
        out_dir,
        tokenizer,
        device,
        report=Report(COLUMNS, on_eval, highlights=HIGHLIGHTS),
        eval_fn=span_metrics,
        resume=resume,
        stop_after=stop_after,
    )


def steps_for(tokens: int, batch_size: int, input_len: int = INPUT_LEN) -> int:
    """Optimizer steps to read ``tokens`` corpus tokens once."""
    return tokens // (batch_size * raw_length(input_len))
