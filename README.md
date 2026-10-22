# Eve

## Layout

Every part of a training run is a component, registered under a kind and a name and picked by
a config value. The trainer is given built components; it never imports a model, dataset or
loss.

```
src/eve/
  registry.py        register(kind, name) · build · names: the component registry
  components.py      the interface of each kind, and the batch keys they share
  experiment.py      experiment configs: layering, --set, config.toml, run()
  data/modules.py    data:      span_corruption, task
  model/eve.py       model:     eve (the architecture; from_config rebuilds it from a checkpoint)
  model/build.py     how a run starts its model: [model] init / preset / config, freeze
  tasks/span.py      loss, metrics: span · report: span_samples
  tasks/questions.py loss, metrics: question, joint · report: task_scores
  train/optim.py     optimizer: adamw · schedule: warmup_cosine
  train/callbacks.py callback:  train_log, evaluate, checkpoint
  train/loop.py      trainer:   default
  train/pretrain.py, train/finetune.py   the experiments `eve pretrain` and `eve finetune` run
configs/
  components/<kind>/<name>.toml   one file per component: its default params
  experiments/*.toml              whole experiments for `eve run`
  pretrain/*.toml, finetune/*.toml   flag presets for `eve pretrain/finetune --config`
```

`eve.components` documents each kind's call signature and the batch keys
(`enc_ids`, `dec_labels`, `tool_pos`, …) a component makes or needs.

## Running experiments

An experiment file names one component per kind and gives each one's params:

```toml
[components]
data = "task"
loss = "joint"          # swap a component: change its name
metrics = "joint"
report = "task_scores"

[model]
init = "artifacts/runs/base-900m/model.pt"

[train]                 # TrainConfig's fields; epochs instead of steps is allowed
epochs = 2.0
lr = 3e-4

[run]
out = "artifacts/runs/joint-base"
```

```bash
uv run eve run configs/experiments/joint-base.toml
```

Params are layered, later ones winning: the component's file in `configs/components/`, the
experiment's table, then `--set section.key=value` (repeatable; values are TOML, bare words
are strings):

```bash
uv run eve run configs/experiments/joint-base.toml --set components.loss=question --set components.metrics=question --set report.values=false --set train.lr=1e-3
```

When you swap a component, also drop or replace params in its table that only the old one
took: an unknown param stops the run with an error rather than being ignored.

`eve pretrain` and `eve finetune` keep their flags and `--config` presets, build the same kind
of experiment, and take `--set` too. Every run writes its resolved experiment to
`<out>/config.toml` and into the `start` record of `metrics.jsonl` and its checkpoints' meta.
To run it again:

```bash
uv run eve run artifacts/runs/joint-base/config.toml --set run.out=artifacts/runs/joint-base-again
```

Seeding happens once per run (`train.seed`), before the model is built. `last.pt` holds the
weights, optimizer and RNG states, and `--resume` (or `run.resume = true`) continues exactly.
Checkpoints store their architecture's registered name, config and tokenizer, so
`load_checkpoint` rebuilds them without knowing the code path that made them.

## Adding a component

1. Write it in a new module. A module in `src/eve/tasks/` is imported automatically; anywhere
   else, add the module to `registry.BUILTIN`.
2. Give it the call signature of its kind in `eve.components`, and put its config params after
   the runtime arguments as keyword arguments with defaults. For example, a loss:

   ```python
   from eve.registry import register

   @register("loss", "weighted_joint")
   def weighted_joint(model, batch, values: float = 1.0) -> dict[str, torch.Tensor]:
       """..."""
   ```

   A model architecture is an `nn.Module` class with a `cfg` (that has `to_dict()`) and a
   `from_config(config)` classmethod.
3. Add `configs/components/<kind>/<name>.toml` with its default params (comments only if it
   has none). `tests/test_components.py` checks that the file exists and that the component
   accepts its keys, then runs a smoke test on tiny input. For a new loss or metric, add the
   data it fits to the test's `DATA_FOR`.
4. Use it: `--set components.loss=weighted_joint --set loss.values=0.5`, or set it in an
   experiment file.
