import argparse
import tomllib
from pathlib import Path
from typing import Any

from eve import defaults

SUFFIX = ".toml"


def config_path(command: str, name: str, root: str | Path = defaults.CONFIG_DIR) -> Path:
    """``base-900m`` -> ``configs/pretrain/base-900m.toml``; a path to a .toml file stays as is."""
    if name.endswith(SUFFIX):
        return Path(name)
    return Path(root) / command / f"{name}{SUFFIX}"


def available(command: str, root: str | Path = defaults.CONFIG_DIR) -> list[str]:
    """The names of the configs ``command`` has in ``root``."""
    return sorted(path.stem for path in (Path(root) / command).glob(f"*{SUFFIX}"))


def read_config(path: Path) -> dict[str, Any]:
    """The options a config file sets, keyed by their argparse dest: ``qk-norm`` -> ``qk_norm``.

    The run directory defaults to ``artifacts/runs/<config name>``; the file may set ``out``.
    """
    if not path.is_file():
        names = ", ".join(available(path.parent.name, path.parent.parent)) or "none"
        raise FileNotFoundError(f"no config {path} (available: {names})")
    with path.open("rb") as file:
        values = tomllib.load(file)
    options = {key.replace("-", "_"): value for key, value in values.items()}
    return {"out": f"{defaults.RUNS_DIR}/{path.stem}", **options}


def apply_config(parser: argparse.ArgumentParser, options: dict[str, Any]) -> None:
    """Make ``options`` the parser's defaults, so flags on the command line still win.

    A key that is not one of the parser's options, or a value outside an option's choices,
    stops the command with a usage error, as a mistyped flag would.
    """
    actions = {action.dest: action for action in parser._actions if action.dest != "config"}
    if unknown := sorted(set(options) - set(actions)):
        parser.error(f"unknown config options: {', '.join(unknown)}")
    for dest, value in options.items():
        choices = actions[dest].choices
        if choices is not None and value not in choices:
            parser.error(f"config option {dest}={value!r} is not one of {sorted(choices)}")
    parser.set_defaults(**options)
