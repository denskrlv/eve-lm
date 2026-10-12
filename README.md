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
uv run pytest
```

Training runs on Apple Silicon (MPS). Inference targets a Raspberry Pi 5 running PyTorch on CPU. Set `COLIBRI_DEVICE=cpu` to force a device.
