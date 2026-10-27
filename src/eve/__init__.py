"""Eve: a tiny function-calling model with Jev-style question heads.

``from eve import Eve, tool`` is the library; ``eve`` on the command line trains and tries it.
The API is imported on first use, so ``import eve.defaults`` (the CLI, the tests) stays cheap.
"""

from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eve.api import (
        Call,
        CheckpointError,
        Eve,
        EveError,
        Intent,
        Param,
        Response,
        ToolBox,
        ToolError,
        ToolLike,
        tool,
    )

try:
    __version__ = metadata.version("eve-lm")
except metadata.PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"

__all__ = [
    "Call",
    "CheckpointError",
    "Eve",
    "EveError",
    "Intent",
    "Param",
    "Response",
    "ToolBox",
    "ToolError",
    "ToolLike",
    "__version__",
    "tool",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import eve.api

        return getattr(eve.api, name)
    raise AttributeError(f"module 'eve' has no attribute {name!r}")
