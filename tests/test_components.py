"""Every registered component: it has a variant file whose keys it takes, it builds from
config, and it works on a tiny input. Then a whole run from an experiment config, and runs
that swap components by config alone."""

import inspect
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_finetune import write_task
from torch import nn

from eve import defaults, registry
from eve.cli import main
from eve.config import preset
from eve.data.render import IGNORE
from eve.experiment import (
    COMPONENTS_DIR,
    CONFIG_FILE,
    Experiment,
    apply_sets,
    read,
    run,
    to_toml,
    trained_meta,
    variant,
)
from eve.model.build import start_model
from eve.model.checkpoint import load_checkpoint, save_checkpoint
from eve.model.eve import Output
from eve.tokenizer.bpe import BYTE0
from eve.train.config import TrainConfig
from eve.train.loop import Report

KIND_ARGS = {  # the runtime arguments before a component's config params, per kind
    "data": 3,  # tokenizer, model_cfg, seed
    "loss": 2,  # model, batch
    "metrics": 2,
    "report": 3,  # data, tokenizer, batch_size
    "optimizer": 3,  # model, cfg, device
    "schedule": 2,  # step, cfg
    "callback": 0,
    "trainer": 0,  # its params are all keywords
}
DATA_FOR = {  # loss/metrics -> the data they read
    "span": "span_corruption", "joint": "task", "retrieval": "retrieval"
}  # fmt: skip
BATCH_KEYS = {  # data -> keys every batch has (the first one's rows are the items)
    "span_corruption": {"enc_ids", "enc_mask", "dec_ids", "dec_mask", "dec_labels"},
    "task": {"enc_ids", "enc_mask", "dec_ids", "dec_mask", "dec_labels"},
    "retrieval": {"q_ids", "q_mask", "t_ids", "t_mask", "positive", "allowed"},
}
ROWS = {"span_corruption": "enc_ids", "task": "enc_ids", "retrieval": "q_ids"}
MODEL_FOR = {"retrieval": "retriever"}  # data -> the model its losses take (default: eve)
TINY = {"preset": "tiny", "max_encoder_len": 128}  # the smallest encoder bucket
TIMING = {"tok_s", "s_per_step"}


def registered() -> list[tuple[str, str]]:
    return [(kind, name) for kind in registry.KINDS for name in registry.names(kind)]


@pytest.mark.parametrize(("kind", "name"), registered())
def test_each_component_has_a_variant_file_it_accepts(kind, name):
    path = Path(COMPONENTS_DIR) / kind / f"{name}.toml"
    assert path.is_file(), f"add {path}"
    if kind == "model":
        target: Any = start_model  # a model's params are how it starts
    else:
        target = registry.get(kind, name)
    params = list(inspect.signature(target).parameters.values())
    takes_any = any(p.kind is p.VAR_KEYWORD for p in params)
    names = {p.name for p in params[KIND_ARGS.get(kind, 0) :]}
    assert takes_any or set(variant(kind, name)) <= names


@pytest.fixture
def span_dir(tmp_path, tokenizer) -> Path:
    rng = np.random.default_rng(0)
    for split, n in [("train", 3000), ("val", 1000)]:
        rng.integers(BYTE0, 600, n).astype(np.uint16).tofile(tmp_path / f"{split}.bin")
    tokenizer.save(tmp_path / "tok.json")
    return tmp_path


@pytest.fixture
def data_dirs(span_dir, library) -> dict[str, Path]:
    write_task(span_dir / "task", library)
    return {"span_corruption": span_dir, "task": span_dir / "task", "retrieval": span_dir / "task"}


def _tiny_config(architecture: str = "eve") -> dict[str, Any]:
    return preset("tiny").to_dict() | ({"embed_dim": 16} if architecture == "retriever" else {})


def tiny_model(architecture: str = "eve") -> nn.Module:
    torch.manual_seed(0)
    return registry.get("model", architecture).from_config(_tiny_config(architecture))


@pytest.fixture
def model():
    return tiny_model()


def build_data(name, data_dirs, tokenizer, library):
    extra = {"library": library} if name in ("task", "retrieval") else {}
    builder = registry.build("data", name, path=str(data_dirs[name]))
    return builder(
        tokenizer, registry.get("model", "eve").from_config(_tiny_config()).cfg, 0, **extra
    )


@pytest.mark.parametrize("name", registry.names("data"))
def test_data_modules_batch_their_items(name, data_dirs, tokenizer, library):
    data = build_data(name, data_dirs, tokenizer, library)
    batch = data.collate([data.train[0], data.train[1]])
    assert BATCH_KEYS[name] <= set(batch)
    assert batch[ROWS[name]].shape[0] == 2
    ids = sum(v.numel() for k, v in batch.items() if k.endswith("_ids"))
    assert 0 < data.batch_tokens(batch) <= ids
    assert len(data.val) > 0


def test_every_loss_and_metric_names_its_data():
    assert set(registry.names("loss")) | set(registry.names("metrics")) <= set(DATA_FOR)


@pytest.mark.parametrize(
    ("kind", "name"), [(k, n) for k in ("loss", "metrics") for n in registry.names(k)]
)
def test_losses_and_metrics_give_finite_scalars(kind, name, data_dirs, tokenizer, library, model):
    data = build_data(DATA_FOR[name], data_dirs, tokenizer, library)
    batch = data.collate([data.train[i] for i in range(3)])
    if DATA_FOR[name] in MODEL_FOR:
        model = tiny_model(MODEL_FOR[DATA_FOR[name]])
    values = registry.build(kind, name)(model, batch)
    assert values and all(v.dim() == 0 and math.isfinite(v.item()) for v in values.values())
    if kind == "loss":
        sum(values.values()).backward()  # trainable: gradients reach the model
        assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", registry.names("model"))
def test_models_rebuild_from_their_config(name):
    model = tiny_model(name)
    rebuilt = registry.get("model", name).from_config(model.cfg.to_dict())
    assert rebuilt.cfg == model.cfg
    ids = torch.full((2, 8), BYTE0)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if name == "retriever":
        assert rebuilt(ids, mask).shape == (2, model.cfg.embed_dim)
        return
    out = rebuilt(ids, mask, ids, mask)
    assert out.lm.shape == (2, 8, model.cfg.vocab_size)


@pytest.mark.parametrize("name", registry.names("report"))
def test_reports_evaluate_a_model(name, data_dirs, tokenizer, library, model):
    data_name = {
        "span_samples": "span_corruption", "task_scores": "task", "retrieval_scores": "retrieval"
    }[name]  # fmt: skip
    data = build_data(data_name, data_dirs, tokenizer, library)
    report = registry.build("report", name)(data, tokenizer, 4)
    assert isinstance(report, Report) and report.on_eval is not None
    if data_name in MODEL_FOR:
        model = tiny_model(MODEL_FOR[data_name])
    assert report.on_eval(model)


@pytest.mark.parametrize("name", registry.names("optimizer"))
def test_optimizers_step(name, model):
    cfg = TrainConfig(steps=10)
    optimizer = registry.build("optimizer", name)(model, cfg, torch.device("cpu"))
    before = model.embeddings.tok.weight.detach().clone()
    model.embeddings.tok.weight.sum().backward()
    optimizer.step()
    assert not torch.equal(before, model.embeddings.tok.weight)


@pytest.mark.parametrize("name", registry.names("schedule"))
def test_schedules_stay_in_range(name):
    cfg = TrainConfig(steps=100, lr=1e-3, warmup_steps=10)
    schedule = registry.build("schedule", name)
    assert all(0 < schedule(step, cfg) <= cfg.lr for step in range(cfg.steps))


@pytest.mark.parametrize("name", registry.names("callback"))
def test_callbacks_have_every_hook(name):
    callback = registry.build("callback", name, **variant("callback", name))()
    assert all(callable(getattr(callback, hook)) for hook in ("on_start", "on_step", "on_end"))


def span_experiment(span_dir: Path, out: Path) -> dict[str, Any]:
    """A 4-step pretraining run on tiny data, as experiment sections."""
    return {
        "components": {
            "data": "span_corruption",
            "loss": "span",
            "metrics": "span",
            "report": "span_samples",
        },
        "model": {**TINY, "tokenizer": str(span_dir / "tok.json"), "dropout": 0.1},
        "data": {"path": str(span_dir)},
        "report": {"samples": ["The capital of France is <x0>."]},
        "train": {
            "steps": 4,
            "batch_size": 2,
            "warmup_steps": 1,
            "eval_every": 2,
            "eval_batches": 2,
            "log_every": 1,
        },
        "run": {"out": str(out), "device": "cpu"},
    }


def results(run_dir: Path) -> list[dict[str, Any]]:
    """A run's logged records without the start record and timing."""
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    return [
        {k: v for k, v in r.items() if k not in TIMING} for r in records if r["kind"] != "start"
    ]


def test_a_run_saves_its_config_and_runs_again_from_it(span_dir, tmp_path):
    first = tmp_path / "first"
    val = run(Experiment.from_dict(span_experiment(span_dir, first)))
    assert {"val_lm", "val_content", "val_content_acc"} <= set(val)
    ckpt = load_checkpoint(first / "model.pt")  # rebuilt through the registry
    assert ckpt.meta["experiment"]["components"]["loss"] == "span"
    start = json.loads((first / "metrics.jsonl").open().readline())
    assert start["experiment"]["train"]["steps"] == 4

    again = tmp_path / "again"
    main(["run", str(first / CONFIG_FILE), "--set", f'run.out="{again}"'])
    assert results(again) == results(first)
    for key, value in ckpt.model.state_dict().items():
        torch.testing.assert_close(
            load_checkpoint(again / "model.pt").model.state_dict()[key], value
        )


def test_epochs_set_the_steps_from_the_train_set(data_dirs, tokenizer, library, tmp_path):
    torch.manual_seed(0)
    init = tmp_path / "pre.pt"
    save_checkpoint(init, registry.get("model", "eve").from_config(_tiny_config()), tokenizer)
    values = {
        "components": {
            "data": "task",
            "loss": "joint",
            "metrics": "joint",
            "report": "task_scores",
        },
        "model": {"init": str(init)},
        "data": {"path": str(data_dirs["task"])},
        "train": {"epochs": 2.0, "batch_size": 3, "warmup_steps": 50, "eval_every": 2},
        "run": {"out": str(tmp_path / "run"), "device": "cpu"},
    }
    run(Experiment.from_dict(values), library=library)
    saved = Experiment.from_dict(tomllib.loads((tmp_path / "run" / CONFIG_FILE).read_text()))
    assert saved.train.steps == 4  # 2 epochs of 6 examples in batches of 3
    assert saved.train.warmup_steps == 4  # capped at the steps
    meta = load_checkpoint(tmp_path / "run" / "model.pt").meta
    assert meta["joint"] is True and meta["init"] == str(init)


EXPERIMENTS = sorted(Path(defaults.CONFIG_DIR, "experiments").glob("*.toml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=[path.stem for path in EXPERIMENTS])
def test_every_experiment_file_loads(path):
    experiment = Experiment.from_dict(read(path))
    for kind in ("data", "loss", "metrics", "report", "model"):
        registry.get(kind, getattr(experiment.components, kind))  # raises if not registered
    assert experiment.run.out == f"{defaults.RUNS_DIR}/{path.stem}"


def test_the_meta_follows_the_components_not_a_stale_table():
    experiments = Path(defaults.CONFIG_DIR, "experiments")
    values = read(experiments / "joint-base.toml")
    assert trained_meta(Experiment.from_dict(values)) == {
        "task": "questions",
        "joint": True,
        "init": "artifacts/runs/base-900m/model.pt",
    }
    moved = Experiment.from_dict(apply_sets(values, ['model.init="other.pt"']))
    assert trained_meta(moved)["init"] == "other.pt"
    pretraining = Experiment.from_dict(read(experiments / "small-250m.toml"))
    assert trained_meta(pretraining) == {"joint": False, "init": "scratch"}


# --- Swapping components by config alone ---------------------------------------------------


def l1_values(model: nn.Module, batch: dict[str, torch.Tensor], weight: float = 1.0):
    """A stand-in loss: the decoder's mean absolute logit on labelled positions."""
    out = model(batch["enc_ids"], batch["enc_mask"], batch["dec_ids"], batch["dec_mask"])
    labelled = batch["dec_labels"] != IGNORE
    return {"l1": weight * out.lm.float().abs()[labelled].mean()}


@dataclass(frozen=True)
class BagConfig:
    vocab_size: int = 8192
    max_encoder_len: int = 128
    d_model: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BagOfWords(nn.Module):
    """A stand-in architecture: mean encoder embedding plus decoder embedding -> logits."""

    def __init__(self, cfg: BagConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BagOfWords":
        return cls(BagConfig(**config))

    def forward(self, enc_ids, enc_mask, dec_ids=None, dec_mask=None, *args) -> Output:
        context = (self.tok(enc_ids) * enc_mask[..., None]).sum(1) / enc_mask.sum(1, keepdim=True)
        return Output(questions=None, lm=self.out(self.tok(dec_ids) + context[:, None]))


def no_report(data, tokenizer, batch_size) -> Report:
    return Report()


@pytest.fixture
def swaps(monkeypatch):
    """Test-only components, registered for one test."""
    for kind, name, component in [
        ("loss", "l1", l1_values),
        ("metrics", "l1", l1_values),
        ("model", "bag", BagOfWords),
        ("report", "none", no_report),
        ("optimizer", "sgd", lambda model, cfg, device, momentum=0.0: torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=momentum)),
    ]:  # fmt: skip
        monkeypatch.setitem(registry._components[kind], name, component)


def run_file(span_dir: Path, tmp_path: Path, *sets: str, model: dict | None = None) -> Path:
    """Write the tiny span experiment (with another ``[model]`` table if given) and run it
    with ``eve run`` and ``sets``."""
    values = span_experiment(span_dir, tmp_path / "run")
    if model is not None:  # another architecture: also no fill-in samples to report
        values |= {"model": model, "report": {}}
    path = tmp_path / "experiment.toml"
    path.write_text(to_toml(values))
    main(["run", str(path), *sets])
    return tmp_path / "run"


def test_swap_the_loss_and_optimizer_by_config(swaps, span_dir, tmp_path):
    out = run_file(
        span_dir, tmp_path,
        "--set", "components.loss=l1", "--set", "components.metrics=l1",
        "--set", "loss.weight=0.5", "--set", "components.optimizer=sgd",
        "--set", "optimizer.momentum=0.9",
    )  # fmt: skip
    records = results(out)
    train = [r for r in records if r["kind"] == "train"]
    assert [r["step"] for r in train] == [1, 2, 3, 4] and all("l1" in r for r in train)
    assert "lm" not in train[0]
    assert {"val_l1"} == {k for r in records if r["kind"] == "val" for k in r} - {"kind", "step"}
    config = Experiment.from_dict(tomllib.loads((out / CONFIG_FILE).read_text()))
    assert config.components.optimizer == "sgd" and config.params["optimizer"] == {"momentum": 0.9}


def test_swap_the_model_by_config(swaps, span_dir, tmp_path):
    config = {"vocab_size": 8192, "max_encoder_len": 128, "d_model": 16}
    model = {"config": config, "tokenizer": str(span_dir / "tok.json")}
    out = run_file(
        span_dir, tmp_path, "--set", "components.model=bag", "--set", "components.report=none",
        model=model,
    )  # fmt: skip
    ckpt = load_checkpoint(out / "model.pt")  # rebuilt as the registered "bag" architecture
    assert isinstance(ckpt.model, BagOfWords)
    losses = [r["lm"] for r in results(out) if r["kind"] == "train"]
    assert len(losses) == 4 and all(math.isfinite(x) for x in losses)


def test_unknown_components_say_what_is_registered(span_dir, tmp_path):
    with pytest.raises(KeyError, match=r"no loss called 'nope'; registered: \['joint'"):
        run(Experiment.from_dict(span_experiment(span_dir, tmp_path) | {"components": {
            "data": "span_corruption", "loss": "nope", "metrics": "span", "report": "span_samples"
        }}))  # fmt: skip
    with pytest.raises(ValueError, match="unknown config sections"):
        Experiment.from_dict({"losses": {}})
    with pytest.raises(TypeError):
        Experiment.from_dict(span_experiment(span_dir, tmp_path) | {"train": {"stpes": 3}})
