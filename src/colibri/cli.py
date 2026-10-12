"""The ``colibri`` command. Later milestones add subcommands (pretrain, finetune, eval, predict)."""

import argparse

import torch

from colibri.config import PRESETS
from colibri.runtime import pick_device


def info(_: argparse.Namespace) -> None:
    print(f"torch {torch.__version__}, device: {pick_device()}")
    print(f"{'preset':<8}{'d_model':>9}{'heads':>7}{'enc':>5}{'dec':>5}{'q':>4}{'params':>10}")
    for name, cfg in PRESETS.items():
        print(
            f"{name:<8}{cfg.d_model:>9}{cfg.n_heads:>7}{cfg.n_encoder_layers:>5}"
            f"{cfg.n_decoder_layers:>5}{cfg.n_question_layers:>4}{cfg.approx_params() / 1e6:>9.1f}M"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="colibri")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="show the device and model presets").set_defaults(run=info)

    args = parser.parse_args(argv)
    args.run(args)
