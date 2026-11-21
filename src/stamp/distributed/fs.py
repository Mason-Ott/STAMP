"""Filesystem helpers for several machines working in one shared folder.

Network storage (NFS) caches directory lookups per machine, including *negative*
ones: a machine that looked for a folder before another machine created it keeps
seeing "no such folder" for a while (tens of seconds), even though creating it
reports "already exists". `refresh_listing` clears that by listing the folder
that contains the stale entry.
"""

import os
import random
import time
from pathlib import Path


def refresh_listing(path: Path) -> None:
    """List the nearest existing folder at or above `path`, which makes this
    machine re-read it and drop stale "does not exist" entries for what another
    machine has created in it since. Errors are ignored: this only refreshes."""
    folder = path
    while True:
        try:
            os.listdir(folder)
            return
        except OSError:
            if folder.parent == folder:
                return
            folder = folder.parent


def ensure_dir(path: Path, attempts: int = 12) -> None:
    """`path.mkdir(parents=True, exist_ok=True)` that survives a race with other
    machines creating the same folders.

    Without the refresh, creating a folder another machine has just created fails
    on network storage with FileExistsError (and the folder still looks missing
    to this machine); creating something inside it fails with FileNotFoundError.
    So after a failure, refresh the stale folder's parent and try again, with a
    short random pause so that machines don't collide again.
    """
    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except (FileExistsError, FileNotFoundError):
            refresh_listing(path)
            if path.is_dir():
                return
            time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))
    path.mkdir(parents=True, exist_ok=True)
