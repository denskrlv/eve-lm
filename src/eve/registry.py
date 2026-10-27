"""Swappable components by kind and name: ``@register("loss", "joint")``, then a config's
``loss = "joint"`` picks it (see ``eve.experiment``).

A component is a function or class. ``build`` binds a config's params to it as keywords; the
caller passes what only exists at run time (the model, a batch, the tokenizer...) when it calls
the result. Each kind's call signature is in ``eve.components``.
"""

import importlib
import importlib.util
import pkgutil
from collections.abc import Callable
from functools import partial
from typing import Any

KINDS = (
    "data", "model", "loss", "metrics", "report", "optimizer", "schedule", "callback", "trainer"
)  # fmt: skip
BUILTIN = {  # kind -> the modules that register its built-in components
    "data": ("eve.data.modules",),
    "model": ("eve.model.eve",),
    "callback": ("eve.train.callbacks",),
    "trainer": ("eve.train.loop",),
    "optimizer": ("eve.train.optim",),
    "schedule": ("eve.train.optim",),
}
DISCOVERED = "eve.tasks"  # every module in this package is imported too: drop a file in it
LIGHT = ("model",)  # kinds inference needs: they never import the training modules

_components: dict[str, dict[str, Any]] = {kind: {} for kind in KINDS}
_loaded: set[str] = set()  # the kinds whose built-ins are imported


def _check_kind(kind: str) -> None:
    if kind not in _components:
        raise KeyError(f"unknown component kind {kind!r}; choose from {list(KINDS)}")


def register[T](kind: str, name: str) -> Callable[[T], T]:
    """Register the decorated function or class as the ``kind`` component called ``name``."""
    _check_kind(kind)

    def add(component: T) -> T:
        if name in _components[kind] and _components[kind][name] is not component:
            raise ValueError(f"a {kind} called {name!r} is already registered")
        _components[kind][name] = component
        return component

    return add


def load_builtins(kind: str | None = None) -> None:
    """Import the modules that register the built-in components of ``kind``, or of every kind
    (once). A light kind (a model) imports only its own modules, so loading a checkpoint for
    inference never pulls in the training code and its dependencies."""
    kinds = [kind] if kind is not None else list(KINDS)
    if not set(kinds) - _loaded:
        return
    _loaded.update(kinds)
    for k in kinds:
        for module in BUILTIN.get(k, ()):
            importlib.import_module(module)
    if all(k in LIGHT for k in kinds) or importlib.util.find_spec(DISCOVERED) is None:
        return
    package = importlib.import_module(DISCOVERED)
    for info in pkgutil.iter_modules(package.__path__, f"{DISCOVERED}."):
        importlib.import_module(info.name)


def get(kind: str, name: str) -> Any:
    """The registered ``kind`` component called ``name``, as registered."""
    _check_kind(kind)
    load_builtins(kind)
    if name not in _components[kind]:
        raise KeyError(f"no {kind} called {name!r}; registered: {names(kind)}")
    return _components[kind][name]


def build(kind: str, name: str, **params: Any) -> Any:
    """The ``kind`` component ``name`` with a config's ``params`` bound (itself without any)."""
    component = get(kind, name)
    return partial(component, **params) if params else component


def names(kind: str) -> list[str]:
    """The names registered for ``kind``, sorted."""
    _check_kind(kind)
    load_builtins(kind)
    return sorted(_components[kind])


def name_of(kind: str, component: Any) -> str:
    """The name ``component`` (a registered function or class) is registered under."""
    load_builtins(kind)
    for name, registered in _components[kind].items():
        if registered is component:
            return name
    raise KeyError(f"{component!r} is not a registered {kind}")
