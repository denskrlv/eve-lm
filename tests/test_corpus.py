import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eve.cli import main
from eve.data.corpus import (
    Mixed,
    encode_corpus,
    iter_documents,
    load_tokens,
    mix_documents,
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


def test_mix_documents_spreads_a_share_of_the_bytes():
    web = [f"web document {i:03d} " + "w" * 80 for i in range(200)]
    extra = [f"tiny {i:02d} " + "t" * 30 for i in range(10)]
    counts = Mixed()
    out = list(mix_documents(web, extra, 0.3, seed=1, counts=counts))
    assert [d for d in out if d.startswith("web")] == web  # web order is kept
    assert counts.share == pytest.approx(0.3, abs=0.01)
    mixed = [d for d in out if d.startswith("tiny")]
    assert len(mixed) == counts.extra_documents > 10  # ran out, so they repeat
    assert sorted(mixed[:10]) == extra  # every one before any repeats
    assert out[0] == web[0] and max(gaps(out)) <= 3  # spread out, not in a block
    assert out == list(mix_documents(web, extra, 0.3, seed=1))
    assert out != list(mix_documents(web, extra, 0.3, seed=2))
    assert list(mix_documents(web, extra, 0.0)) == web
    with pytest.raises(ValueError, match="share"):
        list(mix_documents(web, extra, 1.0))


def gaps(docs):
    """The runs of web documents between two mixed ones."""
    runs, n = [], 0
    for doc in docs:
        if doc.startswith("web"):
            n += 1
        else:
            runs.append(n)
            n = 0
    return runs


def encode_mixed(tokenizer, tmp_path, workers, out="bin"):
    web = [f"Document number {i}: the kitchen lights are on." for i in range(300)]
    extra = [f"Mia asked the speaker to dim lamp {i}." for i in range(20)]
    write_parquet(tmp_path / "web.parquet", web)
    write_parquet(tmp_path / "tinytools.parquet", extra)
    tokenizer.save(tmp_path / "tok.json")
    meta = encode_corpus(
        [tmp_path / "web.parquet"],
        tmp_path / "tok.json",
        tmp_path / out,
        train_tokens=3000,
        val_tokens=500,
        workers=workers,
        docs_per_chunk=8,
        mix=[tmp_path / "tinytools.parquet"],
        mix_share=0.25,
    )
    return web, meta


def test_encode_corpus_mixes_into_train_only(tokenizer, tmp_path):
    web, meta = encode_mixed(tokenizer, tmp_path, workers=2)
    assert meta["train_tokens"] == 3000
    assert meta["mix"]["documents"] == 20 and meta["mix"]["share"] == 0.25
    assert meta["mix"]["byte_share"] == pytest.approx(0.25, abs=0.03)

    val = load_tokens(tmp_path / "bin/val.bin").tolist()
    expected = []
    for doc in web:
        expected += tokenizer.encode(doc) + [EOS]
    assert val == expected[: len(val)]  # web text only, in order
    train = tokenizer.decode(load_tokens(tmp_path / "bin/train.bin").tolist())
    assert "Mia asked the speaker" in train and "Document number" in train

    # the read-ahead past val.bin goes back into train, so workers don't change the output
    encode_mixed(tokenizer, tmp_path, workers=1, out="bin1")
    for name in ("val.bin", "train.bin"):
        assert (tmp_path / "bin" / name).read_bytes() == (tmp_path / "bin1" / name).read_bytes()


def test_eve_corpus_encode_mix(tokenizer, tmp_path, capsys):
    write_parquet(tmp_path / "web.parquet", [f"Document {i}: lights on." for i in range(200)])
    write_parquet(tmp_path / "tt.parquet", [f"Turn the lamp to {i}." for i in range(50)])
    tokenizer.save(tmp_path / "tok.json")
    args = ["corpus", "encode", "--parquet", str(tmp_path / "web.parquet"), "--tokenizer",
            str(tmp_path / "tok.json"), "--out", str(tmp_path / "bin"), "--train-tokens", "1000",
            "--val-tokens", "200", "--workers", "1"]  # fmt: skip
    main([*args, "--mix", str(tmp_path / "tt.parquet"), "--mix-share", "0.2"])
    assert "mixed in from 50 documents" in capsys.readouterr().out
    assert json.loads((tmp_path / "bin/meta.json").read_text())["mix"]["share"] == 0.2
    with pytest.raises(SystemExit, match="mix-share"):
        main([*args, "--mix", str(tmp_path / "tt.parquet"), "--mix-share", "1"])
