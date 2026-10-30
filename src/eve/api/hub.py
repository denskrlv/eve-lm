import re
from pathlib import Path

from eve.api.errors import CheckpointError
from eve.model.checkpoint import MODEL_FILE, RETRIEVER_FILE

REPO_ID = re.compile(r"[\w.-]+/[\w.-]+")  # "user/eve-32m"


def resolve(source: str | Path, revision: str | None = None) -> Path:
    """The local checkpoint file for ``source``: a ``.pt`` file, a directory holding
    ``model.pt``, or a Hugging Face Hub repo id, downloaded once and then read from the Hub's
    cache. ``revision`` (a branch, tag or commit) is for Hub repos only."""
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / MODEL_FILE
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
        return Path(hf_hub_download(str(source), MODEL_FILE, revision=revision))
    except Exception as error:  # the Hub raises many kinds: offline, no repo, no access
        raise CheckpointError(
            f"no local file {source}, and the Hub download failed: {error}"
        ) from error


def companion(
    source: str | Path, revision: str | None = None, filename: str = RETRIEVER_FILE
) -> Path | None:
    """``filename`` beside the checkpoint ``source`` names (in its directory or Hub repo), or
    None if there is none or it can't be fetched: a companion is optional."""
    path = Path(source).expanduser()
    if path.is_file():
        path = path.parent
    if path.is_dir():
        return path / filename if (path / filename).is_file() else None
    if not REPO_ID.fullmatch(str(source)):
        return None
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(str(source), filename, revision=revision))
    except Exception:  # no huggingface_hub, offline, or the repo has no such file
        return None
