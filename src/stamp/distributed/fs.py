"""Filesystem helpers for several machines working in one shared folder."""

import random
import time
from pathlib import Path


def ensure_dir(path: Path, attempts: int = 12) -> None:
    """`path.mkdir(parents=True, exist_ok=True)` that survives a race with other
    machines creating the same folders.

    On network storage, creating a folder another machine is creating at that
    moment can fail with FileExistsError even with `exist_ok` (the check that it
    is a folder can see a stale cache), or with FileNotFoundError (a parent was
    created and is not visible yet). Both pass within moments, so retry with a
    random pause, so that the machines don't collide again.
    """
    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except (FileExistsError, FileNotFoundError):
            time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))
            if path.is_dir():
                return
    path.mkdir(parents=True, exist_ok=True)
