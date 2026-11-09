"""Download, verify and cache dataset files."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class DataIntegrityError(RuntimeError):
    """A downloaded or cached file does not match its registered checksum."""


@dataclass(frozen=True)
class DataFile:
    """A remote file pinned by URL and sha256."""

    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    @property
    def cache_name(self) -> str:
        return f"{self.sha256[:16]}-{self.filename}"


def cache_dir() -> Path:
    """``$DCMBENCH_DATA_DIR`` if set, otherwise ``~/.cache/dcmbench``."""
    configured = os.environ.get("DCMBENCH_DATA_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "dcmbench"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(file: DataFile) -> Path:
    """
    Return a local path to ``file``, downloading it into the cache if needed.

    The checksum is verified on download and on every use of the cached copy.
    A corrupt cached copy is downloaded again once; a mismatching download raises.
    """
    directory = cache_dir()
    path = directory / file.cache_name
    if path.exists():
        if sha256_of(path) == file.sha256:
            return path
        path.unlink()

    directory.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=directory, prefix=".download-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(file.url, timeout=60) as response:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
        actual = sha256_of(tmp)
        if actual != file.sha256:
            raise DataIntegrityError(
                f"{file.url} has sha256 {actual}, expected {file.sha256}. "
                "The remote file changed or the download was corrupted."
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
