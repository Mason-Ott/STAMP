"""Runs a step over its items, coordinating with other machines through files.

Every output file `X` can have two siblings:
  X.lock    someone is producing X right now (kept fresh by a heartbeat thread)
  X.failed  producing X raised an error (contains the traceback)

Several machines can run the same command against a shared filesystem. Each one
skips outputs that exist, are locked by someone else, or have failed, and does
the rest. An item whose inputs are still being produced elsewhere waits for them.

Steps themselves know nothing about any of this: they receive input paths that
exist and write one output path.
"""

import logging
import os
import random
import socket
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from tqdm import tqdm

from stamp.distributed.config import CoordinationConfig
from stamp.distributed.fs import ensure_dir

_logger = logging.getLogger("stamp")

T = TypeVar("T")


@dataclass(frozen=True)
class Step(Generic[T]):
    """A pipeline step as the scheduler sees it: one output file per item."""

    name: str
    items: Sequence[T]
    output_of: Callable[[T], Path]
    inputs_of: Callable[[T], Sequence[Path]]
    """Files from earlier steps that `run` reads for this item (may be empty)."""
    run: Callable[[T, list[Path], Path], None]
    """(item, inputs that exist, output path) -> writes the output."""
    allow_missing_inputs: bool = False
    """If True, `run` gets whichever inputs exist (at least one) instead of failing."""
    later_outputs_of: Callable[[T], Sequence[Path]] = lambda item: ()
    """Outputs of later steps built from this item's output. If any exists, the
    item counts as done even if its own output has since been deleted."""
    delete_inputs_when_done: bool = False
    """Delete the item's inputs once its output is written (intermediate files)."""

    def is_done(self, item: T) -> bool:
        if self.output_of(item).exists():
            return True
        return any(path.exists() for path in self.later_outputs_of(item))


@dataclass
class StepReport:
    done: int = 0
    already_done: int = 0
    busy_elsewhere: int = 0
    failed: int = 0
    failed_earlier: int = 0
    missing_inputs: int = 0
    """Items skipped because an input doesn't exist (no .failed marker is left,
    so they run once the inputs appear, e.g. after an earlier step is run)."""
    failures: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if nothing failed and nothing was left waiting for inputs."""
        return not (self.failed or self.missing_inputs)

    def summary(self) -> str:
        counts = {k: v for k, v in vars(self).items() if isinstance(v, int) and v}
        return ", ".join(f"{k}={v}" for k, v in counts.items()) or "nothing to do"


class MissingInputsError(RuntimeError):
    pass


class InputState(Enum):
    READY = "ready"
    FAILED = "failed"
    MISSING = "missing"


def lock_path(output: Path) -> Path:
    return output.with_name(output.name + ".lock")


def failed_path(output: Path) -> Path:
    return output.with_name(output.name + ".failed")


def run_step(step: Step[T], coordination: CoordinationConfig) -> StepReport:
    """Produce every missing output of `step` that no other machine is working on.

    Items are visited in a random order per process, so machines started at the
    same time don't all queue up behind the same item. Items whose inputs all
    exist come first: an item still waiting for an input that another machine is
    producing is left for last instead of blocking the ones that can run now.
    """
    rng = random.Random(f"{socket.gethostname()}-{os.getpid()}")
    ready, waiting = [], []
    for item in step.items:
        inputs = step.inputs_of(item)
        (ready if all(path.exists() for path in inputs) else waiting).append(item)
    rng.shuffle(ready)
    rng.shuffle(waiting)
    items = ready + waiting

    report = StepReport()
    progress = tqdm(items, desc=step.name)
    for item in progress:
        output = step.output_of(item)
        progress.set_postfix_str(output.name, refresh=False)

        if step.is_done(item):
            report.already_done += 1
            continue
        if failed_path(output).exists() and not coordination.retry_failed:
            report.failed_earlier += 1
            continue

        with claim(output, coordination.lock_stale_after_s) as owned:
            if not owned:
                report.busy_elsewhere += 1
            elif step.is_done(item):  # another machine finished it after our check
                report.already_done += 1
            else:
                outcome = _run_item(step, item, output, coordination)
                if outcome is _Outcome.DONE:
                    report.done += 1
                elif outcome is _Outcome.MISSING_INPUTS:
                    report.missing_inputs += 1
                else:
                    report.failed += 1
                    report.failures.append(output)

    _logger.info(f"{step.name}: {report.summary()}")
    if report.missing_inputs:
        _logger.warning(
            f"{step.name}: {report.missing_inputs} items have missing inputs; "
            "run the earlier steps first (or check why they failed)"
        )
    if report.failed_earlier:
        _logger.warning(
            f"{step.name}: {report.failed_earlier} items failed in an earlier run; "
            "set pipeline.coordination.retry_failed: true or delete their .failed "
            "files to retry"
        )
    return report


class _Outcome(Enum):
    DONE = "done"
    FAILED = "failed"
    MISSING_INPUTS = "missing inputs"


def _run_item(
    step: Step[T], item: T, output: Path, coordination: CoordinationConfig
) -> _Outcome:
    """Run one item. On error, record a .failed marker.

    Missing inputs are not this item's failure (usually an earlier step hasn't
    run, or failed for that input), so they leave no marker: the item runs as
    soon as its inputs exist.
    """
    failed_path(output).unlink(missing_ok=True)
    try:
        inputs = _gather_inputs(step, item, coordination)
    except MissingInputsError as error:
        _logger.warning(f"{step.name}: skipping {output.name}: {error}")
        return _Outcome.MISSING_INPUTS
    try:
        step.run(item, inputs, output)
    except Exception:
        _logger.exception(f"{step.name}: failed to produce {output}")
        failed_path(output).write_text(traceback.format_exc())
        return _Outcome.FAILED
    if step.delete_inputs_when_done:
        # Only now that the output is safely written: until then, the inputs are
        # what a rerun would recover from.
        for path in inputs:
            path.unlink(missing_ok=True)
    return _Outcome.DONE


def _gather_inputs(
    step: Step[T], item: T, coordination: CoordinationConfig
) -> list[Path]:
    expected = list(step.inputs_of(item))
    states = {
        path: wait_for(
            path,
            stale_after_s=coordination.lock_stale_after_s,
            poll_interval_s=coordination.poll_interval_s,
        )
        for path in expected
    }
    ready = [path for path, state in states.items() if state is InputState.READY]
    unavailable = {p: s for p, s in states.items() if s is not InputState.READY}

    if unavailable:
        details = ", ".join(f"{p} ({s.value})" for p, s in unavailable.items())
        if not step.allow_missing_inputs or not ready:
            raise MissingInputsError(f"inputs not available: {details}")
        _logger.warning(f"{step.name}: continuing without {details}")
    return ready


def wait_for(path: Path, *, stale_after_s: float, poll_interval_s: float) -> InputState:
    """Return once `path` exists, has failed, or nobody is producing it."""
    announced = False
    while True:
        if path.exists():
            return InputState.READY
        if failed_path(path).exists():
            return InputState.FAILED
        if not _is_locked(path, stale_after_s):
            return InputState.MISSING
        if not announced:
            _logger.info(f"waiting for {path} (being produced elsewhere)")
            announced = True
        time.sleep(poll_interval_s)


@contextmanager
def claim(output: Path, stale_after_s: float) -> Iterator[bool]:
    """Try to lock `output`. Yields True if this process now owns it.

    While owned, a background thread refreshes the lock's modification time, so
    other machines can tell a slow job from a dead one.
    """
    lock = lock_path(output)
    if not _create_lock(lock, stale_after_s):
        yield False
        return

    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_refresh_until_stopped,
        args=(lock, stop, max(1.0, min(60.0, stale_after_s / 5))),
        daemon=True,
    )
    heartbeat.start()
    try:
        yield True
    finally:
        stop.set()
        heartbeat.join()
        lock.unlink(missing_ok=True)


def _create_lock(lock: Path, stale_after_s: float) -> bool:
    """Atomically create `lock`, taking over a stale one. False if someone holds it."""
    ensure_dir(lock.parent)
    age = _lock_age_s(lock)
    if age is not None and age > stale_after_s:
        _logger.warning(f"taking over stale lock {lock}")
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{socket.gethostname()} pid={os.getpid()} t={time.time():.0f}\n")
    return True


def _refresh_until_stopped(
    lock: Path, stop: threading.Event, interval_s: float
) -> None:
    while not stop.wait(interval_s):
        try:
            os.utime(lock)
        except FileNotFoundError:
            return


def lock_holder(output: Path, stale_after_s: float) -> str | None:
    """Host name of whoever is producing `output` right now, or None."""
    if not _is_locked(output, stale_after_s):
        return None
    try:
        return lock_path(output).read_text().split(maxsplit=1)[0]
    except (FileNotFoundError, IndexError):
        return "?"


def _is_locked(output: Path, stale_after_s: float) -> bool:
    """True if someone holds a lock on `output` that isn't stale."""
    age = _lock_age_s(lock_path(output))
    return age is not None and age <= stale_after_s


def _lock_age_s(lock: Path) -> float | None:
    """Seconds since the lock was last refreshed, or None if there is no lock."""
    try:
        return time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return None
