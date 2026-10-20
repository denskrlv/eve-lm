import json
import logging
import os
from collections.abc import Iterable, Iterator
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from eve import defaults
from eve.data.schema import load_library
from eve.tokenizer.bpe import EOS, BPETokenizer


def iter_documents(paths: Iterable[str | Path]) -> Iterator[str]:
    """The ``text`` column of each parquet file, one document at a time."""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=PARQUET_BATCH_ROWS, columns=["text"]
        ):
            yield from batch.column(0).to_pylist()


def take_bytes(docs: Iterable[str], n_bytes: int, skip_bytes: int = 0) -> Iterator[str]:
    """Skip about ``skip_bytes`` of UTF-8 text, then yield documents until ``n_bytes`` more."""
    seen = 0
    for doc in docs:
        size = len(doc.encode("utf-8"))
        if seen >= skip_bytes + n_bytes:
            return
        if seen >= skip_bytes:
            yield doc
        seen += size


def tokenizer_texts(
    corpus: Iterable[str],
    tools_dir: str | Path = defaults.TOOLS_DIR,
    commands_file: str | Path | None = defaults.COMMANDS_FILE,
    weight: int = defaults.TOKENIZER_WEIGHT,
) -> Iterator[tuple[str, int]]:
    """Tokenizer training text: the corpus, plus every tool schema and sample command.

    The schemas and commands are a few KB next to ~50 MB of web text, so they count ``weight``
    times. Each starts with a space, as the renderer writes them after a special token.
    """
    for doc in corpus:
        yield doc, 1
    for tool in load_library(tools_dir).values():
        yield tool.render(), weight
    if commands_file is not None and Path(commands_file).exists():
        for line in Path(commands_file).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                yield " " + line.strip(), weight


# ---- encoding with several processes --------------------------------------------------------

# A few chunks per worker at a time, so we never read far ahead of what we need.
CHUNKS_PER_WORKER = 4
_tokenizer: BPETokenizer | None = None
logger = logging.getLogger(__name__)
PARQUET_BATCH_ROWS = 1024  # rows read from a parquet file at a time


def _init_worker(tokenizer_path: str) -> None:
    global _tokenizer
    _tokenizer = BPETokenizer.load(tokenizer_path)


def _encode_docs(docs: list[str]) -> np.ndarray:
    assert _tokenizer is not None, "_init_worker runs first in each worker"
    ids: list[int] = []
    for doc in docs:
        ids.extend(_tokenizer.encode(doc))
        ids.append(EOS)
    return np.array(ids, dtype=np.uint16)


def _chunks(docs: Iterable[str], size: int) -> Iterator[list[str]]:
    it = iter(docs)
    while chunk := list(islice(it, size)):
        yield chunk


def _encode_splits(
    paths: list[str | Path],
    tokenizer_path: str | Path,
    out_dir: Path,
    train_tokens: int,
    val_tokens: int,
    workers: int,
    docs_per_chunk: int,
) -> dict[str, int]:
    """Encode documents with ``workers`` processes into ``val.bin``, then ``train.bin``.
    Returns the tokens written to each."""
    counts = {"val": 0, "train": 0}
    chunks = _chunks(iter_documents(paths), docs_per_chunk)
    bar = tqdm(total=val_tokens + train_tokens, unit="tok", unit_scale=True, desc="encoding")
    with (
        Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool,
        open(out_dir / "val.bin", "wb") as val_file,
        open(out_dir / "train.bin", "wb") as train_file,
    ):
        files = {"val": val_file, "train": train_file}
        while counts["train"] < train_tokens:
            in_flight = list(islice(chunks, workers * CHUNKS_PER_WORKER))
            if not in_flight:
                break
            for ids in pool.imap(_encode_docs, in_flight):
                split = "val" if counts["val"] < val_tokens else "train"
                if split == "train":
                    ids = ids[: train_tokens - counts["train"]]
                ids.tofile(files[split])
                counts[split] += len(ids)
                bar.update(len(ids))
                if counts["train"] >= train_tokens:
                    break
    bar.close()
    return counts


def encode_corpus(
    paths: Iterable[str | Path],
    tokenizer_path: str | Path,
    out_dir: str | Path,
    train_tokens: int,
    val_tokens: int,
    workers: int | None = None,
    docs_per_chunk: int = 256,
) -> dict[str, Any]:
    """Encode documents into ``val.bin`` (whole documents, at least ``val_tokens``) and then
    ``train.bin`` (exactly ``train_tokens``). Returns what it writes to ``meta.json``."""
    paths = list(paths)
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary too large for uint16")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    counts = _encode_splits(
        paths, tokenizer_path, out_dir, train_tokens, val_tokens, workers, docs_per_chunk
    )

    meta = {
        "tokenizer": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "val_tokens": counts["val"],
        "train_tokens": counts["train"],
        "sources": [str(p) for p in paths],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if counts["train"] < train_tokens:
        logger.warning(f"the corpus ran out after {counts['train']:,} train tokens")
    return meta


def load_tokens(path: str | Path) -> np.ndarray:
    """A token file as a read-only memmap (nothing is read until it is indexed)."""
    return np.memmap(path, dtype=np.uint16, mode="r")
