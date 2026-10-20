# Eve

A tiny local model for function calling, built from scratch as a learning project.

You give it a command and 1–5 tool schemas ("turn off the kitchen lights" plus `set_lights`, `set_thermostat`, …). It does two things:

- **Needle part:** writes the call(s) with their arguments.
- **Jev part:** answers a fixed set of questions in one pass, each with a calibrated probability:
  - should it call a tool, ask the user, say the request is unsupported, or do nothing?
  - which tools, in which order?
  - how sure is each argument value?

The build follows the roadmap in Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

| Milestone | Book box | What |
|---|---|---|
| M0 | – | Project setup, model presets |
| M1 | 1 Data preparation & sampling | Tokenizer, pretraining corpus, span corruption, task renderer |
| M2 | 2 Attention | Self, causal and cross attention |
| M3 | 3 Architecture | Encoder, value decoder, question block |
| M4 | 4–6 Pretraining, training loop, evaluation | Span-corruption pretraining |
| M5 | 7 Load pretrained weights | Checkpoints |
| M6 | Stage 3 datasets | Tool library and task data |
| M7 | 8 Classifier fine-tuning | Jev-style question heads |
| M8 | 9 Instruction fine-tuning | Value decoder, joint run |
| M9 | 6 Evaluation | Metrics and calibration |
| M10 | – | Inference CLI, Raspberry Pi 5 |

## Setup

```bash
uv sync
uv run eve info
uv run eve model   # parameters per part of each preset (M3)
uv run pytest
```

## Data (M1)

Tools live in `data/tools/*.json` (format in `src/eve/data/schema.py`). The pretraining corpus is one FineWeb-Edu shard:

```bash
mkdir -p data/raw/fineweb-edu
curl -L -o data/raw/fineweb-edu/000_00000.parquet \
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet
uv run eve tokenizer train   # 8k byte-level BPE -> artifacts/tokenizer/eve-8k.json
uv run eve tokenizer stats   # tokens per word and sample splits
uv run eve corpus            # 250M train + 2M val tokens -> data/bin/*.bin
```

## Task data (M6)

Tools for training live in `data/tools/*.json` with request templates; `data/tools/heldout/` holds two
domains (math, travel) the model never trains on. Code adds ~260 generated tools so the model learns to
read schemas it hasn't seen.

External data, converted to our format with held-out topics (math, travel) removed:
[xLAM function-calling 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
(gated: accept its terms on Hugging Face first; research use) and
[MASSIVE](https://github.com/alexa/massive) en-US (CC BY 4.0).

```bash
uvx --from huggingface_hub hf download Salesforce/xlam-function-calling-60k --repo-type dataset --local-dir data/raw/xlam
mkdir -p data/raw/massive && curl -L https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz \
  | tar -xz -C data/raw/massive --include '*/en-US.jsonl'
uv run eve data external                 # -> data/external/*.jsonl
```

```bash
uv run eve data build                    # templates + chat-written + external -> data/task/*.jsonl
uv run eve data batch --split val        # scenarios for a chat to write requests for
uv run eve data check batch-0001         # validate what the chat wrote
```

How chat-written requests are made: `data/generation/README.md`.

## Pretraining (M4)

T5 span corruption over `data/bin/train.bin`: one pass over 250M tokens is ~13.7k steps, about 2.6 h on an M3 Pro (eager; `--compile` is ~2x faster).

Training recipes are named configs in `configs/<command>/<name>.toml`. Their keys are the command's flags (`qk-norm = true`), they become the defaults, and flags on the command line still win. The run writes to `artifacts/runs/<name>` unless the config or `--out` says otherwise.

```bash
uv run eve pretrain --config small-250m  # -> artifacts/runs/small-250m/{metrics.jsonl,last.pt,model.pt}
uv run eve pretrain --config base-900m   # ~32M parameters on 850M tokens (data/bin-900m)
uv run eve pretrain --config base-900m --resume          # continue from last.pt after an interruption
uv run eve pretrain --config base-900m --lr 2e-3 --stop-after 2000 --out artifacts/runs/ab-lr2e-3
```

Each eval logs the val loss and fill-in-the-blank samples (`The capital of France is [Paris].`). `val_content` is the loss on the words inside the holes only (sentinels and `<eos>` left out), the number to watch. Fine-tuning starts from `model.pt` with `load_pretrained` (M5).

```bash
uv run eve fill "Paris is the capital of <x0>."     # prompt the pretrained model
uv run eve compare artifacts/runs/ab-*              # val metrics of several runs side by side
```

A/B runs use the full run's schedule and stop early (`--stop-after`), so each arm is the start of a real run and the winner continues with `--resume`.

Training runs on Apple Silicon (MPS). Inference targets a Raspberry Pi 5 running PyTorch on CPU. Set `EVE_DEVICE=cpu` to force a device.

## Question heads (M7)

Fine-tune the Jev heads (intent, and the ordered "what is call #i?" slots) from the pretrained checkpoint. Book ch. 6 trains only the question block with the body frozen first, then everything. Since v3, one full fine-tune straight from the pretrained model does as well:

```bash
uv run eve finetune --config questions-small     # v3 recipe from the small pretrained model
uv run eve finetune --config questions-base      # the same from base-900m
uv run eve finetune --init artifacts/runs/ab-dropout0-rope-qk/model.pt --freeze --out artifacts/runs/questions-frozen
uv run eve finetune --init artifacts/runs/questions-frozen/model.pt --out artifacts/runs/questions-full
uv run eve finetune --scratch small --out artifacts/runs/questions-scratch   # E1: what pretraining buys
```

Each eval logs val loss and accuracy per question (`tools_acc` = right tools in the right order, `exact_acc` = that and the right intent), prints what the heads answer for one val example per intent, and the accuracy on tools from the held-out domains. Val is mostly MASSIVE, where 84% of examples are CALL, so `intent_acc` starts near 84% by always answering CALL.

Score a finished model on whole task files: recall per intent and their mean (macro), `tools` and `exact` as the evals log them, and a table of which intent each gold intent got. By default it scores val, the held-out-tool templates and MASSIVE test; val and held-out come out as the run's last eval logged them.

```bash
uv run eve evaluate artifacts/runs/questions-full-v3                  # a run directory or a model.pt
uv run eve evaluate artifacts/runs/questions-full-v3 --sets heldout-templates --by-tool
uv run eve evaluate artifacts/runs/questions-full-v3 --sets probe-candidates --json
```

`--by-tool` adds a row per tool in the gold calls (CALLs, and ASKs that name the tool), worst exact accuracy first, with which intents its examples got. An example with calls to two tools counts for both.

Ask a fine-tuned model about your own requests (tool names from `data/tools/*.json`, or `--schema file.json` for your own schemas):

```bash
uv run eve ask --tools set_lights,play_music "turn off the kitchen lights"
uv run eve ask --tools set_lights,play_music        # no request: type them one by one
```
