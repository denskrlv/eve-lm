import json
import logging
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eve.cli import main
from eve.config import preset
from eve.data import corpus
from eve.data.corpus import (
    encode_corpus,
    iter_documents,
    load_tokens,
    mix_documents,
    take_bytes,
    tokenizer_texts,
)
from eve.data.modules import span_corruption
from eve.tokenizer.bpe import EOS, BPETokenizer

ROOT = Path(__file__).parents[1]


def write_parquet(path, docs):
    pq.write_table(pa.table({"text": docs, "id": [str(i) for i in range(len(docs))]}), path)


def test_take_bytes():
    docs = ["aaaa", "bbbb", "cccc", "dddd"]
    assert list(take_bytes(docs, 8)) == ["aaaa", "bbbb"]
    assert list(take_bytes(docs, 4, skip_bytes=8)) == ["cccc"]


def test_mix_documents_balances_bytes_and_stops_with_the_shortest_source():
    long, short = ["x" * 10] * 5, ["y"] * 100
    mixed = list(mix_documents([long, short]))
    assert mixed[:12] == ["x" * 10] + ["y"] * 10 + ["x" * 10]
    assert mixed.count("x" * 10) == 5 and mixed.count("y") == 50  # stops when x runs out
    assert list(mix_documents([["a", "b"]])) == ["a", "b"]


def test_tokenizer_texts_add_weighted_schemas():
    texts = list(
        tokenizer_texts(["web text"], ROOT / "data/tools", ROOT / "data/tokenizer/commands.txt", 7)
    )
    assert texts[0] == ("web text", 1)
    assert any(t.startswith("<tool> set lights") and w == 7 for t, w in texts)
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
        [paths],
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


def test_encode_corpus_logs_progress_when_the_bar_is_off(tokenizer, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(corpus, "LOG_EVERY", 0.0)  # a line per chunk instead of per minute
    log = logging.getLogger("eve.data.corpus")
    log.addHandler(caplog.handler)  # eve's loggers may not propagate after main() ran
    monkeypatch.setattr(log, "level", logging.INFO)
    write_parquet(tmp_path / "web.parquet", [f"Document {i}: lights on." for i in range(300)])
    tokenizer.save(tmp_path / "tok.json")
    try:
        encode_corpus(
            [[tmp_path / "web.parquet"]],
            tmp_path / "tok.json",
            tmp_path / "bin",
            train_tokens=3000,
            val_tokens=500,
            workers=1,
            docs_per_chunk=8,
        )
    finally:
        log.removeHandler(caplog.handler)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("encoding")]
    assert len(lines) > 3
    assert re.fullmatch(
        r"encoding [\d.,]+M/[\d.,]+M tokens \(\d+%\) \| [\d.]+M tok/s \| eta \S+", lines[-1]
    )
    assert "(100%)" in lines[-1]


def test_tokenizer_train_and_stats_commands(tmp_path, capsys):
    docs = [("The kitchen lights are on. Turn them off at night, please. " * 20)] * 8
    write_parquet(tmp_path / "web.parquet", docs)
    tokenizer = str(tmp_path / "tok.json")
    web = ["--parquet", str(tmp_path / "*.parquet")]

    main(["tokenizer", "train", *web, "--text", "2k", "--vocab-size", "600", "--out", tokenizer])
    assert Path(tokenizer).is_file()

    main(["tokenizer", "stats", *web, "--tokenizer", tokenizer, "--text", "1k", "--skip", "0"])
    out = capsys.readouterr().out
    assert "tokens/word" in out and "vocabulary" in out


def test_pretraining_refuses_token_files_from_another_tokenizer(tokenizer, tmp_path):
    write_parquet(tmp_path / "web.parquet", [f"Document {i}: lights on." for i in range(300)])
    tokenizer.save(tmp_path / "tok.json")
    meta = encode_corpus(
        [[tmp_path / "web.parquet"]], tmp_path / "tok.json", tmp_path / "bin", 3000, 500, workers=1
    )
    assert meta["tokenizer_fingerprint"] == tokenizer.fingerprint()
    span_corruption(tokenizer, preset("tiny"), 0, path=str(tmp_path / "bin"))  # its own: fine

    other = BPETokenizer.train([("some other text entirely", 3)], vocab_size=400)
    with pytest.raises(ValueError, match="eve corpus encode"):
        span_corruption(other, preset("tiny"), 0, path=str(tmp_path / "bin"))
