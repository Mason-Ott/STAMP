"""Small file-system helpers shared by the steps."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py

from stamp.distributed.fs import ensure_dir


@contextmanager
def atomic_h5_file(path: Path) -> Iterator[h5py.File]:
    """Open an HDF5 file for writing that only appears at `path` once complete.

    Data goes to a hidden temporary file in the same folder, which is renamed to
    `path` on success and deleted on failure. Other machines therefore never see
    a half-written output, and `path.exists()` means "finished".
    """
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(tmp, "w") as f:
            yield f
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
