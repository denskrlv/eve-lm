from pathlib import Path

import pytest
import torch

import eve
from eve import Eve
from eve.api.toolbox import BM25, Dense, ToolBox
from eve.config import preset
from eve.data.render import Call, TaskExample
from eve.data.schema import ASK_VALUE, Tool
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Eve as EveModel
from eve.model.retriever import Retriever, RetrieverConfig, padded, query_ids, tool_ids
from eve.tasks.retrieval import (
    Case,
    RetrievalData,
    _Table,
    cases,
    measure,
    pairs,
    retrieval_loss,
    retrieval_metrics,
)
from eve.tokenizer.bpe import EOS, BPETokenizer

HOME = ["set_lights", "set_thermostat", "play_music", "lock_door", "set_timer", "add_to_list"]


def tiny(**overrides: int) -> Retriever:
    torch.manual_seed(0)
    return Retriever(RetrieverConfig(**preset("tiny").to_dict() | {"embed_dim": 16} | overrides))


def example(library: dict[str, Tool], text: str, gold: list[str], others: list[str]) -> TaskExample:
    tools = tuple(library[name] for name in gold + others)
    calls = tuple(  # ASK, with every required argument missing
        Call(name, {a.name: ASK_VALUE for a in library[name].args if a.required}) for name in gold
    )
    return TaskExample(text, tools, "ask", calls)


class Oracle:
    """Scores the tools named in a request highest."""

    def scores(self, text: str, tools: list[Tool]) -> list[float]:
        return [float(tool.name in text) for tool in tools]


# ---- the model --------------------------------------------------------------------------------


def test_embeddings_have_length_one_and_ignore_padding(
    library: dict[str, Tool], tokenizer: BPETokenizer
) -> None:
    model = tiny().eval()
    short = query_ids(tokenizer, "lights off", 64)
    long = query_ids(tokenizer, "turn the kitchen lights off and play some jazz please", 64)
    ids, mask = padded([short, long])
    both = model(ids, mask)
    alone = model(*padded([short]))
    assert torch.allclose(both.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(both[0], alone[0], atol=1e-5)  # padding changes nothing
    tools = model.embed_tools([library[n] for n in HOME], tokenizer)
    assert tools.shape == (len(HOME), 16)


def test_tool_schemas_are_cut_at_max_tool_len(
    library: dict[str, Tool], tokenizer: BPETokenizer
) -> None:
    ids = tool_ids(tokenizer, library["set_thermostat"], 8)
    assert len(ids) == 8 and ids[-1] == EOS


def test_start_copies_the_encoder_of_an_eve_checkpoint(
    tmp_path: Path, tokenizer: BPETokenizer
) -> None:
    torch.manual_seed(0)
    body = EveModel(preset("tiny"))
    save_checkpoint(tmp_path / "eve.pt", body, tokenizer)
    started = Retriever.start(tmp_path / "eve.pt", embed_dim=8)
    model = started.model
    assert model.cfg.embed_dim == 8 and model.project.out_features == 8
    copied = model.state_dict()
    assert all(
        torch.equal(copied[k], v) for k, v in body.state_dict().items() if k.startswith("encoder")
    )
    with pytest.raises(ValueError, match="d_model"):
        Retriever.start(tmp_path / "eve.pt", d_model=32)

    save_checkpoint(tmp_path / "retriever.pt", model, tokenizer, {"task": "retrieval"})
    again = Retriever.start(tmp_path / "retriever.pt")
    assert torch.equal(again.model.project.weight, model.project.weight)  # all of it loads
    with pytest.raises(ValueError, match="embed_dim"):
        Retriever.start(tmp_path / "retriever.pt", embed_dim=4)
    assert isinstance(load_checkpoint(tmp_path / "retriever.pt").model, Retriever)


# ---- training ---------------------------------------------------------------------------------


def test_a_tool_that_serves_the_request_is_never_its_negative(
    library: dict[str, Tool], tokenizer: BPETokenizer
) -> None:
    table = _Table(tokenizer, tiny().cfg)
    rows = [
        example(library, "lights off and play jazz", ["set_lights", "play_music"], ["lock_door"]),
        example(library, "set a timer", ["set_timer"], ["set_thermostat", "lock_door"]),
    ]
    items = pairs(rows, table)
    assert len(items) == 3  # one pair per correct tool
    groups = [tool.name for tool in table.index]
    data = RetrievalData(items, items, table.ids, groups, {}, hard=1)
    batch = data.collate(items)
    names = list(table.index)
    columns = [
        names[i].name
        for i in dict.fromkeys(t for p in items for t in (p.positive, *p.negatives[:1]))
    ]
    assert columns == ["set_lights", "lock_door", "play_music", "set_timer", "set_thermostat"]
    assert batch["t_ids"].shape == (6, 128) and batch["q_ids"].shape == (3, 128)  # one shape
    assert not batch["allowed"][:, 5:].any()  # lock_door joins once; the fillers never
    for row, pair in enumerate(items):
        own = names[pair.positive].name
        assert columns[batch["positive"][row]] == own
        blocked = {columns[c] for c in range(len(columns)) if not batch["allowed"][row, c]}
        other = (
            {"set_lights", "play_music"} - {own} if own in ("set_lights", "play_music") else set()
        )
        assert blocked == other  # the request's other correct tool only

    model = tiny()
    loss = retrieval_loss(model, batch)["contrastive"]
    loss.backward()
    assert torch.isfinite(loss) and model.log_scale.grad is not None
    assert 0.0 <= retrieval_metrics(model, batch)["top1_acc"].item() <= 1.0
    assert data.collate(items[:1]).keys() == batch.keys()


def test_same_capability_tools_are_not_negatives(
    library: dict[str, Tool], tokenizer: BPETokenizer
) -> None:
    table = _Table(tokenizer, tiny().cfg)
    items = pairs([example(library, "lights off", ["set_lights"], ["lock_door"])], table)
    groups = [
        "light" if tool.name in ("set_lights", "lock_door") else tool.name for tool in table.index
    ]
    batch = RetrievalData(items, items, table.ids, groups, {}, hard=1).collate(items)
    assert batch["allowed"][:, :2].tolist() == [[True, False]]


# ---- evaluation -------------------------------------------------------------------------------


def test_cases_hold_the_correct_tools_in_registries_of_a_size(library: dict[str, Tool]) -> None:
    rows = [example(library, f"use {n}", [n], [m for m in HOME if m != n][:2]) for n in HOME]
    rows.append(TaskExample("hello", (library["set_timer"],), "none", ()))
    made = cases(rows, 4, seed=1)
    assert len(made) == len(HOME)  # NONE examples name no tools
    assert all(len(c.tools) == 4 and c.gold <= {t.name for t in c.tools} for c in made)
    assert made == cases(rows, 4, seed=1)
    assert all(len(c.tools) == len(HOME) for c in cases(rows, None))
    scores = measure(Oracle(), made)
    assert scores == {"all@5": 1.0, "recall@5": 1.0, "mrr": 1.0}
    assert measure(Oracle(), made, top=1)["all@1"] == 1.0


def test_similar_registries_fill_up_with_the_same_domain_first(library: dict[str, Tool]) -> None:
    rows = [example(library, f"use {n}", [n], []) for n in HOME]  # no distractors of their own
    lights = [c for c in cases(rows, 3, seed=0, fill="similar") if c.gold == {"set_lights"}]
    assert {t.name for t in lights[0].tools} == {"set_lights", "set_thermostat", "lock_door"}
    randoms = [
        {t.name for t in c.tools}
        for s in range(5)
        for c in cases(rows, 3, s)
        if c.text == "use set_lights"
    ]
    assert any(names != {"set_lights", "set_thermostat", "lock_door"} for names in randoms)
    with pytest.raises(ValueError, match="fill"):
        cases(rows, 3, fill="nearest")


def test_measure_counts_all_and_partial_hits(library: dict[str, Tool]) -> None:
    tools = tuple(library[n] for n in HOME)
    case = Case("use set_lights", frozenset({"set_lights", "play_music"}), tools)
    scores = measure(Oracle(), [case], top=1)
    assert scores == {"all@1": 0.0, "recall@1": 0.5, "mrr": 1.0}


# ---- scorers and Eve.load ---------------------------------------------------------------------


def test_dense_scores_are_cosines_and_cached(
    library: dict[str, Tool], tokenizer: BPETokenizer
) -> None:
    model = tiny()
    dense = Dense(model, tokenizer)
    tools = [library[n] for n in HOME]
    scores = dense.scores("lights off", tools)
    expected = (
        model.embed_tools(tools, tokenizer) @ model.embed_queries(["lights off"], tokenizer)[0]
    )
    assert scores == pytest.approx(expected.tolist(), abs=1e-5)
    assert set(dense._tools) == set(tools)
    box = ToolBox(tools, scorer=dense)
    assert len(box.search("lights off", 5)) == 5 and model.training  # the model's mode is kept


def save_retriever(path: Path, tokenizer: BPETokenizer) -> None:
    save_checkpoint(path, tiny(), tokenizer, {"task": "retrieval"})


def test_load_finds_a_retriever_beside_the_model(tmp_path: Path, tokenizer: BPETokenizer) -> None:
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "model.pt", EveModel(preset("tiny")), tokenizer, {"joint": True})
    assert isinstance(Eve.load(tmp_path).tools.scorer, BM25)  # none there

    save_retriever(tmp_path / "retriever.pt", tokenizer)
    loaded = Eve.load(tmp_path / "model.pt")
    assert isinstance(loaded.tools.scorer, Dense) and loaded.info["scorer"] == "Dense"
    assert not loaded.tools.scorer.retriever.training  # type: ignore[attr-defined]
    assert isinstance(Eve.load(tmp_path, retriever=None).tools.scorer, BM25)
    with pytest.raises(eve.CheckpointError, match="isn't a retriever"):
        Eve.load(tmp_path, retriever=tmp_path / "model.pt")
