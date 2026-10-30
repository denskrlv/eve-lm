"""Downloading the TinyTools sources into ``data/raw``: plain files, files from Hugging Face
dataset repos, and the wanted members of a zip or tar archive."""

import io
import logging
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
TIMEOUT = 60  # seconds without data before a download fails


@dataclass(frozen=True)
class Url:
    """A file at ``url``, saved as ``path`` (relative to the raw folder)."""

    url: str
    path: str


@dataclass(frozen=True)
class Hub:
    """A file of a Hugging Face dataset repo, saved as ``path``."""

    repo: str
    filename: str
    path: str


@dataclass(frozen=True)
class Archive:
    """The members of a zip or .tar.gz under ``url`` that start with one of ``keep`` (after the
    archive's top folder, which is dropped), saved under ``path``."""

    url: str
    keep: tuple[str, ...]
    path: str


Download = Url | Hub | Archive


def _target(download: Download, raw: Path) -> Path:
    return raw / download.path


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "eve-lm"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data: bytes = response.read()
    return data


def _save(path: Path, data: bytes) -> None:
    """Write through a temporary name, so a stopped download leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _extract(data: bytes, download: Archive, out: Path) -> int:
    """Save the wanted members; return how many. Paths are checked to stay inside ``out``."""
    members: list[tuple[str, bytes]] = []
    if download.url.endswith(".zip") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members.append((info.filename, archive.read(info)))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                if member.isfile() and (f := archive.extractfile(member)) is not None:
                    members.append((member.name, f.read()))
    count = 0
    for name, content in members:
        inner = name.split("/", 1)[1] if "/" in name else name
        if not inner.startswith(download.keep):
            continue
        path = (out / inner).resolve()
        if not path.is_relative_to(out.resolve()):
            raise ValueError(f"archive member outside the folder: {name}")
        _save(path, content)
        count += 1
    return count


def fetch(downloads: tuple[Download, ...], raw: Path, force: bool = False) -> list[str]:
    """Download what isn't there yet (everything with ``force``); return what was saved."""
    saved = []
    for download in downloads:
        target = _target(download, raw)
        if target.exists() and not force:
            continue
        if isinstance(download, Hub):
            from huggingface_hub import hf_hub_download

            logger.info(f"downloading {download.repo}/{download.filename}")
            staging = target.parent / ".hub"  # not the shared cache: the files are large
            got = hf_hub_download(
                download.repo, download.filename, repo_type="dataset", local_dir=staging
            )
            Path(got).replace(target)
            shutil.rmtree(staging)
        elif isinstance(download, Archive):
            logger.info(f"downloading {download.url}")
            count = _extract(_get(download.url), download, target)
            if not count:
                raise ValueError(f"nothing under {download.keep} in {download.url}")
        else:
            logger.info(f"downloading {download.url}")
            _save(target, _get(download.url))
        saved.append(download.path)
    return saved
