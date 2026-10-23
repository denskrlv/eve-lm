import json
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, islice
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow.parquet as pq

from eve import defaults
from eve.data.schema import load_library
from eve.progress import Bar
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


@dataclass
class Mixed:
    """What ``mix_documents`` has yielded so far."""

    web_bytes: int = 0
    extra_bytes: int = 0
    extra_documents: int = 0

    @property
    def share(self) -> float:
        """The share of the bytes that came from the extra documents."""
        total = self.web_bytes + self.extra_bytes
        return self.extra_bytes / total if total else 0.0


def mix_documents(
    web: Iterable[str],
    extra: Sequence[str],
    share: float,
    seed: int = 0,
    counts: Mixed | None = None,
) -> Iterator[str]:
    """``web`` with ``extra`` spread through it, so the extra documents make about ``share`` of
    the UTF-8 bytes. The extra documents come in a shuffled order; when they run out they are
    shuffled again and repeat. ``counts`` (if given) is updated as documents are yielded."""
    if not 0 <= share < 1:
        raise ValueError(f"share must be in [0, 1), got {share}")
    counts = counts if counts is not None else Mixed()
    rng = np.random.default_rng(seed)
    queue: list[int] = []
    for doc in web:
        counts.web_bytes += len(doc.encode("utf-8"))
        yield doc
        while extra and counts.extra_bytes < share * (counts.web_bytes + counts.extra_bytes):
            if not queue:
                queue = rng.permutation(len(extra)).tolist()
            doc = extra[queue.pop()]
            counts.extra_bytes += len(doc.encode("utf-8"))
            counts.extra_documents += 1
            yield doc


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


def _encode_into(
    pool: PoolType,
    docs: Iterator[str],
    file: BinaryIO,
    limit: int,
    whole_documents: bool,
    workers: int,
    docs_per_chunk: int,
    bar: Bar,
) -> tuple[int, list[str]]:
    """Encode ``docs`` into ``file`` until it holds ``limit`` tokens: whole documents (the
    last chunk may go past ``limit``) or exactly ``limit``. Returns the tokens written and the
    documents read ahead but not written, in order."""
    count = 0
    chunks = _chunks(docs, docs_per_chunk)
    while count < limit:
        in_flight = list(islice(chunks, workers * CHUNKS_PER_WORKER))
        if not in_flight:
            break
        for i, ids in enumerate(pool.imap(_encode_docs, in_flight)):
            if count >= limit:
                return count, [doc for chunk in in_flight[i:] for doc in chunk]
            if not whole_documents:
                ids = ids[: limit - count]
            ids.tofile(file)
            count += len(ids)
            bar.update(len(ids))
    return count, []


def _encode_splits(
    paths: list[str | Path],
    tokenizer_path: str | Path,
    out_dir: Path,
    train_tokens: int,
    val_tokens: int,
    workers: int,
    docs_per_chunk: int,
    mix: tuple[Sequence[str], float, int, Mixed] | None = None,
) -> dict[str, int]:
    """Encode documents with ``workers`` processes into ``val.bin``, then ``train.bin``.
    ``mix`` (documents, share, seed, counts) is spread through the train documents only, so
    the val tokens stay comparable between runs with and without it. Returns the tokens
    written to each."""
    web = iter_documents(paths)
    bar = Bar(val_tokens + train_tokens, "encoding", "tokens")
    with (
        Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool,
        open(out_dir / "val.bin", "wb") as val_file,
        open(out_dir / "train.bin", "wb") as train_file,
    ):
        val, ahead = _encode_into(
            pool, web, val_file, val_tokens, True, workers, docs_per_chunk, bar
        )
        docs: Iterator[str] = chain(ahead, web)
        if mix is not None:
            extra, share, seed, counts = mix
            docs = mix_documents(docs, extra, share, seed, counts)
        train, _ = _encode_into(
            pool, docs, train_file, train_tokens, False, workers, docs_per_chunk, bar
        )
    bar.close()
    return {"val": val, "train": train}


def encode_corpus(
    paths: Iterable[str | Path],
    tokenizer_path: str | Path,
    out_dir: str | Path,
    train_tokens: int,
    val_tokens: int,
    workers: int | None = None,
    docs_per_chunk: int = 256,
    mix: Iterable[str | Path] = (),
    mix_share: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Encode documents into ``val.bin`` (whole documents, at least ``val_tokens``) and then
    ``train.bin`` (exactly ``train_tokens``). Returns what it writes to ``meta.json``.

    ``mix`` are parquet files (TinyTools) spread through the train documents to make about
    ``mix_share`` of their text; ``val.bin`` stays web text only."""
    paths, mix = list(paths), list(mix)
    extra = list(iter_documents(mix))
    if mix and not extra:
        raise ValueError(f"no documents in {[str(p) for p in mix]}")
    mixed = Mixed()
    tokenizer = BPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary too large for uint16")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    counts = _encode_splits(
        paths,
        tokenizer_path,
        out_dir,
        train_tokens,
        val_tokens,
        workers,
        docs_per_chunk,
        (extra, mix_share, seed, mixed) if extra else None,
    )

    meta = {
        "tokenizer": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "val_tokens": counts["val"],
        "train_tokens": counts["train"],
        "sources": [str(p) for p in paths],
    }
    if extra:
        # read ahead of train.bin's end, so the counts are a little high
        passes = mixed.extra_documents / len(extra)
        meta["mix"] = {
            "sources": [str(p) for p in mix],
            "share": mix_share,
            "byte_share": round(mixed.share, 4),
            "documents": len(extra),
            "passes": round(passes, 2),
        }
        if passes > 1:
            logger.warning(
                f"the {len(extra):,} mixed documents repeat about {passes:.1f} times; "
                "write more or lower --mix-share"
            )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if counts["train"] < train_tokens:
        logger.warning(f"the corpus ran out after {counts['train']:,} train tokens")
    return meta


def load_tokens(path: str | Path) -> np.ndarray:
    """A token file as a read-only memmap (nothing is read until it is indexed)."""
    return np.memmap(path, dtype=np.uint16, mode="r")
