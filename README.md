# colibri-lm

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
uv run colibri info
uv run colibri model   # parameters per part of each preset (M3)
uv run pytest
```

## Data (M1)

Tools live in `data/tools/*.json` (format in `src/colibri/data/schema.py`). The pretraining corpus is one FineWeb-Edu shard:

```bash
mkdir -p data/raw/fineweb-edu
curl -L -o data/raw/fineweb-edu/000_00000.parquet \
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet
uv run colibri tokenizer train   # 8k byte-level BPE -> artifacts/tokenizer/colibri-8k.json
uv run colibri tokenizer stats   # tokens per word and sample splits
uv run colibri corpus            # 250M train + 2M val tokens -> data/bin/*.bin
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
uv run colibri data external                 # -> data/external/*.jsonl
```

```bash
uv run colibri data build                    # templates + chat-written + external -> data/task/*.jsonl
uv run colibri data batch --split val        # scenarios for a chat to write requests for
uv run colibri data check batch-0001         # validate what the chat wrote
```

How chat-written requests are made: `data/generation/README.md`.

## Pretraining (M4)

T5 span corruption over `data/bin/train.bin`: one pass over 250M tokens is ~13.7k steps, about 2.6 h on an M3 Pro.

```bash
uv run colibri pretrain                      # -> artifacts/runs/pretrain-small/{metrics.jsonl,last.pt,model.pt}
uv run colibri pretrain --resume             # continue from last.pt after an interruption
```

Each eval logs the val loss and fill-in-the-blank samples (`The capital of France is [Paris].`). Fine-tuning starts from `model.pt` with `load_pretrained` (M5).

Training runs on Apple Silicon (MPS). Inference targets a Raspberry Pi 5 running PyTorch on CPU. Set `COLIBRI_DEVICE=cpu` to force a device.
