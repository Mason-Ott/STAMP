"""Noticing when someone else logs into a shared machine, so a worker can leave."""

import _thread
import getpass
import logging
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_logger = logging.getLogger("stamp")


def other_users() -> list[str]:
    """Users other than the current one with a login session here (from `who`)."""
    result = subprocess.run(["who"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        _logger.warning(f"`who` failed: {result.stderr.strip()}")
        return []
    me = getpass.getuser()
    users = {line.split()[0] for line in result.stdout.splitlines() if line.strip()}
    return sorted(users - {me})


class PresenceWatch:
    """Background check that interrupts the main thread when another user appears.

    The interrupt arrives as KeyboardInterrupt, the same as Ctrl+C, so the
    pipeline unwinds normally: locks are released and half-written files removed.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.intruders: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    @property
    def triggered(self) -> bool:
        return bool(self.intruders)

    def _watch(self) -> None:
        while not self._stop.wait(self.interval_s):
            if users := other_users():
                self.intruders = users
                _thread.interrupt_main()
                return


@contextmanager
def yield_to_other_users(interval_s: float) -> Iterator[PresenceWatch]:
    watch = PresenceWatch(interval_s)
    watch._thread.start()
    try:
        yield watch
    finally:
        watch._stop.set()
