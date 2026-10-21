"""Experiments: one config that picks and parameterizes every component of a run.

An experiment file (TOML) has a ``[components]`` table naming one registered component per
kind, a table of params per kind, the ``[train]`` schedule (``TrainConfig``'s fields), and
where the run writes (``[run]``)::

    [components]
    data = "task"
    loss = "joint"          # swap a component: change its name here
    [model]
    init = "artifacts/runs/base-900m/model.pt"
    [train]
    epochs = 2.0            # or steps
    lr = 3e-4
    [run]
    out = "artifacts/runs/joint-base"

Params are layered, later ones winning: the component's variant file
``configs/components/<kind>/<name>.toml``, then the experiment's ``[<kind>]`` table, then
``--set kind.key=value`` overrides. Every run writes the resolved experiment to
``<out>/config.toml`` (``eve run`` runs it again) and into its checkpoints' meta.
"""

import json
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from eve import defaults, registry
from eve.model.build import start_model
from eve.runtime import pick_device, seed_everything
from eve.train.config import TrainConfig

COMPONENTS_DIR = f"{defaults.CONFIG_DIR}/components"  # variant files: <kind>/<name>.toml
CONFIG_FILE = "config.toml"  # the resolved experiment, in every run directory
PARAM_KINDS = ("model", "data", "loss", "metrics", "report", "optimizer", "schedule", "trainer")
SECTIONS = ("components", *PARAM_KINDS, "train", "run", "meta")


@dataclass(frozen=True)
class Components:
    """The registered name of each component a run uses."""

    data: str
    loss: str
    metrics: str
    report: str
    model: str = "eve"
    optimizer: str = "adamw"
    schedule: str = "warmup_cosine"
    trainer: str = "default"
    callbacks: tuple[str, ...] = ("train_log", "evaluate", "checkpoint")


@dataclass(frozen=True)
class RunConfig:
    """Where a run writes and how it starts: a new run or ``resume``, and an early stop."""

    out: str = defaults.FINETUNE_RUN
    device: str | None = None  # default: the fastest available (runtime.pick_device)
    resume: bool = False
    stop_after: int | None = None


@dataclass(frozen=True)
class Experiment:
    """A whole run, resolved: the components, their params, the schedule and the run."""

    components: Components
    params: dict[str, dict[str, Any]]  # kind -> params, for each of PARAM_KINDS
    train: TrainConfig
    run: RunConfig = RunConfig()
    meta: dict[str, Any] = field(default_factory=dict)  # added to the checkpoints' meta
    epochs: float | None = None  # if set, run() makes train.steps this many train-set passes

    def to_dict(self) -> dict[str, Any]:
        """The experiment as plain values, in the shape ``from_dict`` reads."""
        components = asdict(self.components)
        components["callbacks"] = list(self.components.callbacks)
        epochs = {} if self.epochs is None else {"epochs": self.epochs}  # steps not known yet
        return {
            "components": components,
            **{kind: dict(self.params.get(kind, {})) for kind in PARAM_KINDS},
            "train": self.train.to_dict() | epochs,
            "run": asdict(self.run),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(
        cls, values: dict[str, Any], variants: str | Path = COMPONENTS_DIR
    ) -> "Experiment":
        """An experiment from a config's sections, each kind's params on top of its variant
        file. ``[train]`` may give ``epochs`` instead of ``steps``: run() then counts the steps
        from the train set's size."""
        if unknown := sorted(set(values) - set(SECTIONS)):
            raise ValueError(f"unknown config sections {unknown}; choose from {list(SECTIONS)}")
        table = dict(values.get("components", {}))
        if "callbacks" in table:
            table["callbacks"] = tuple(table["callbacks"])
        components = Components(**table)
        params = {
            kind: variant(kind, getattr(components, kind), variants) | values.get(kind, {})
            for kind in PARAM_KINDS
        }
        train = dict(values.get("train", {}))
        epochs = train.pop("epochs", None)
        if epochs is not None:
            train.setdefault("steps", 1)  # a placeholder until run() knows the train set
        return cls(
            components,
            params,
            TrainConfig(**train),
            RunConfig(**values.get("run", {})),
            dict(values.get("meta", {})),
            epochs,
        )


def variant(kind: str, name: str, root: str | Path = COMPONENTS_DIR) -> dict[str, Any]:
    """The default params of component ``name`` from its variant file (none if it has none)."""
    path = Path(root) / kind / f"{name}.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as file:
        return tomllib.load(file)


def merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``top`` over ``base``: tables are merged key by key, any other value replaced."""
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_set(assignment: str) -> dict[str, Any]:
    """``"train.lr=1e-3"`` -> ``{"train": {"lr": 0.001}}``. The value is read as TOML
    (numbers, booleans, ``["a", "b"]``, quoted strings), else taken as a bare string."""
    path, sep, raw = assignment.partition("=")
    keys = [key.strip() for key in path.split(".")]
    if not sep or len(keys) < 2 or not all(keys):
        raise ValueError(f"--set wants section.key=value, got {assignment!r}")
    try:
        value: Any = tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw
    out: dict[str, Any] = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        out = {key: out}
    return out


def apply_sets(values: dict[str, Any], assignments: Sequence[str]) -> dict[str, Any]:
    """``values`` with every ``--set`` assignment applied, in order."""
    for assignment in assignments:
        values = merge(values, parse_set(assignment))
    return values


def read(path: str | Path) -> dict[str, Any]:
    """The sections of an experiment file."""
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # a TOML basic string
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"can't write {type(value).__name__} to TOML: {value!r}")


def to_toml(values: dict[str, Any], prefix: str = "") -> str:
    """Tables of plain values as TOML (``None`` values left out: TOML has no null)."""
    plain = {k: v for k, v in values.items() if v is not None and not isinstance(v, dict)}
    tables = {k: v for k, v in values.items() if isinstance(v, dict)}
    lines = [f"{k} = {_toml_value(v)}" for k, v in plain.items()]
    text = "\n".join(lines) + "\n" if lines else ""
    for key, table in tables.items():
        name = f"{prefix}{key}"
        text += f"\n[{name}]\n" + to_toml(table, f"{name}.")
    return text


def with_epochs(train: TrainConfig, epochs: float, n_train: int) -> TrainConfig:
    """``train`` with the steps to read ``n_train`` items ``epochs`` times, and the warmup
    capped at that, as the command line does."""
    steps = round(epochs * n_train / train.batch_size)
    return replace(train, steps=steps, warmup_steps=min(train.warmup_steps, steps))


def run(experiment: Experiment, **data_args: Any) -> dict[str, float]:
    """Build every component of ``experiment`` from the registry and train. ``data_args`` go
    to the data builder as they are (e.g. a ``library`` of tools already loaded). Returns the
    last val metrics.

    Order, which the RNG depends on: seed, start the model (random weights come from the
    seeded RNG), then the data, the report and the rest.
    """
    c, params = experiment.components, experiment.params
    train_cfg = experiment.train
    seed = train_cfg.seed
    seed_everything(seed)
    started = start_model(architecture=c.model, **params["model"])
    model_cfg = getattr(started.model, "cfg")  # noqa: B009 (nn.Module hides it from mypy)
    data = registry.build("data", c.data, **params["data"])(
        started.tokenizer, model_cfg, seed, **data_args
    )
    if experiment.epochs is not None:
        train_cfg = with_epochs(train_cfg, experiment.epochs, len(data.train))
    resolved = Experiment(c, params, train_cfg, experiment.run, experiment.meta)
    out_dir = Path(experiment.run.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG_FILE).write_text(to_toml(resolved.to_dict()))

    trainer = registry.build("trainer", c.trainer, **params["trainer"])
    meta = {**experiment.meta, "trainable": started.trainable, "experiment": resolved.to_dict()}
    val: dict[str, float] = trainer(
        started.model,
        registry.build("loss", c.loss, **params["loss"]),
        data,
        train_cfg,
        out_dir,
        started.tokenizer,
        pick_device(experiment.run.device),
        report=registry.build("report", c.report, **params["report"])(
            data, started.tokenizer, train_cfg.batch_size
        ),
        eval_fn=registry.build("metrics", c.metrics, **params["metrics"]),
        resume=experiment.run.resume,
        stop_after=experiment.run.stop_after,
        meta=meta,
        optimizer=registry.build("optimizer", c.optimizer, **params["optimizer"]),
        schedule=registry.build("schedule", c.schedule, **params["schedule"]),
        callbacks=[
            registry.build("callback", name, **variant("callback", name))() for name in c.callbacks
        ],
        experiment=resolved.to_dict(),
    )
    return val
