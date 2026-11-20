"""Running a job as one of several workers on a shared machine.

Adds two things around the job:
  * courtesy on shared machines: don't start while someone else is logged in,
    and stop cleanly if someone logs in later;
  * clean stopping: SIGINT and SIGTERM (e.g. from `stamp workers stop`) raise
    KeyboardInterrupt, so the job unwinds normally. With the workqueue this
    releases the worker's locks, leaving its unfinished item to the others.
"""

import logging
import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext

from stamp.distributed.config import WorkersConfig
from stamp.distributed.presence import other_users, yield_to_other_users

_logger = logging.getLogger("stamp")

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_STOPPED = 130  # conventional exit code for "interrupted"


def run_as_worker(job: Callable[[], bool], workers: WorkersConfig | None) -> int:
    """Run `job` (which returns True on success). Returns the process exit code.

    Leaving because another user logged in is not an error (exit code 0): the
    other workers carry on with the remaining work.
    """
    watch_users = workers is None or workers.yield_to_other_users
    interval_s = workers.presence_check_s if workers is not None else 60.0

    if watch_users and (present := other_users()):
        _logger.warning(f"not starting: other users are logged in ({present})")
        return EXIT_OK

    watch_context = yield_to_other_users(interval_s) if watch_users else nullcontext()
    with _interrupt_on_stop_signals(), watch_context as watch:
        try:
            succeeded = job()
        except KeyboardInterrupt:
            if watch is not None and watch.triggered:
                _logger.warning(
                    f"stopped: other users logged in ({watch.intruders}); "
                    "unfinished work is left for the other workers"
                )
                return EXIT_OK
            _logger.warning(
                "stopped on request; unfinished work is left for the others"
            )
            return EXIT_STOPPED

    if not succeeded:
        return EXIT_FAILURES
    _logger.info("finished")
    return EXIT_OK


@contextmanager
def _interrupt_on_stop_signals() -> Iterator[None]:
    """Make SIGINT and SIGTERM raise KeyboardInterrupt, however the process was started.

    Needed because a process started in the background by a non-interactive
    shell inherits SIGINT as *ignored*, in which case Python never raises
    KeyboardInterrupt and the presence check's interrupt does nothing.
    """
    stop_signals = (signal.SIGINT, signal.SIGTERM)
    previous = {
        sig: signal.signal(sig, _raise_keyboard_interrupt) for sig in stop_signals
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)
