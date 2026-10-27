import re
from pathlib import Path

from eve.api.errors import CheckpointError

FILENAME = "model.pt"  # the checkpoint in a model directory or Hub repo
REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")  # "user/eve-joint-base"


def resolve(source: str | Path, revision: str | None = None) -> Path:
    """The local checkpoint file for ``source``: a ``.pt`` file, a directory holding
    ``model.pt``, or a Hugging Face Hub repo id, downloaded once and then read from the Hub's
    cache. ``revision`` (a branch, tag or commit) is for Hub repos only."""
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / FILENAME
    if path.is_file():
        return path
    if path.exists() or not REPO_ID.fullmatch(str(source)):
        raise CheckpointError(f"no checkpoint at {source}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise CheckpointError(
            f"{source} isn't a local file; loading from the Hub needs huggingface_hub"
        ) from None
    try:
        return Path(hf_hub_download(str(source), FILENAME, revision=revision))
    except Exception as error:  # the Hub raises many kinds: offline, no repo, no access
        raise CheckpointError(
            f"no local file {source}, and the Hub download failed: {error}"
        ) from error
