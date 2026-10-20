import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eve.data.corpus import (
    encode_corpus,
    iter_documents,
    load_tokens,
    take_bytes,
    tokenizer_texts,
)
from eve.tokenizer.bpe import EOS

ROOT = Path(__file__).parents[1]


def write_parquet(path, docs):
    pq.write_table(pa.table({"text": docs, "id": [str(i) for i in range(len(docs))]}), path)


def test_take_bytes():
    docs = ["aaaa", "bbbb", "cccc", "dddd"]
    assert list(take_bytes(docs, 8)) == ["aaaa", "bbbb"]
    assert list(take_bytes(docs, 4, skip_bytes=8)) == ["cccc"]


def test_tokenizer_texts_add_weighted_schemas():
    texts = list(
        tokenizer_texts(["web text"], ROOT / "data/tools", ROOT / "data/tokenizer/commands.txt", 7)
    )
    assert texts[0] == ("web text", 1)
    assert any(t.startswith("<tool> set_lights") and w == 7 for t, w in texts)
    assert (" turn off the kitchen lights", 7) in texts
    assert not any(t.startswith(" #") for t, _ in texts)


def test_encode_corpus(tokenizer, tmp_path):
    docs = [f"Document number {i}: the kitchen lights are on." for i in range(300)]
    write_parquet(tmp_path / "a.parquet", docs[:150])
    write_parquet(tmp_path / "b.parquet", docs[150:])
    paths = sorted(tmp_path.glob("*.parquet"))
    assert list(iter_documents(paths)) == docs
    tokenizer.save(tmp_path / "tok.json")

    meta = encode_corpus(
        paths,
        tmp_path / "tok.json",
        tmp_path / "bin",
        train_tokens=3000,
        val_tokens=500,
        workers=2,
        docs_per_chunk=16,
    )
    assert meta["train_tokens"] == 3000 and meta["val_tokens"] >= 500
    assert json.loads((tmp_path / "bin/meta.json").read_text()) == meta

    val, train = load_tokens(tmp_path / "bin/val.bin"), load_tokens(tmp_path / "bin/train.bin")
    assert len(train) == 3000 and val[-1] == EOS  # val ends on a document boundary
    stream = np.concatenate([val, train]).tolist()
    expected = []
    for doc in docs:
        expected += tokenizer.encode(doc) + [EOS]
    assert stream == expected[: len(stream)]
