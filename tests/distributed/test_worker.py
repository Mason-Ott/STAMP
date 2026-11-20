"""run_as_worker: leaving shared machines and stopping cleanly."""

import os
import signal
import threading
import time
from pathlib import Path

import pytest

from stamp.distributed import presence, worker
from stamp.distributed.config import CoordinationConfig, WorkersConfig
from stamp.distributed.workqueue import Step, failed_path, lock_path, run_step

FAST = CoordinationConfig(lock_stale_after_s=60.0, poll_interval_s=0.05)
WORKERS = WorkersConfig(hosts=["here"], presence_check_s=0.1)


def blocked_job(tmp_path: Path) -> tuple[object, Path]:
    """A job whose only item waits for an input that another machine 'holds'.

    Returns the job and the output it holds a lock on while waiting.
    """
    held_input = tmp_path / "input"
    lock_path(held_input).write_text("elsewhere pid=1")
    output = tmp_path / "output"
    step = Step(
        name="blocked",
        items=["item"],
        output_of=lambda item: output,
        inputs_of=lambda item: [held_input],
        run=lambda item, inputs, out: out.write_text("done"),
    )

    def job() -> bool:
        return run_step(step, FAST).failed == 0

    return job, output


def test_leaves_cleanly_when_another_user_logs_in(tmp_path, monkeypatch):
    job, held = blocked_job(tmp_path)
    users = iter([[], [], ["bob"]])  # nobody at first, then bob logs in
    monkeypatch.setattr(presence, "other_users", lambda: next(users, ["bob"]))
    monkeypatch.setattr(worker, "other_users", lambda: [])

    started = time.time()
    assert worker.run_as_worker(job, WORKERS) == worker.EXIT_OK
    assert time.time() - started < 10
    assert not lock_path(held).exists(), "lock released for the other workers"
    assert not failed_path(held).exists(), "leaving is not a failure"


def test_does_not_start_while_someone_else_is_logged_in(monkeypatch):
    monkeypatch.setattr(worker, "other_users", lambda: ["bob"])

    def job() -> bool:
        raise AssertionError("should not run")

    assert worker.run_as_worker(job, WORKERS) == worker.EXIT_OK


def test_ignores_other_users_when_told_to(monkeypatch):
    monkeypatch.setattr(worker, "other_users", lambda: ["bob"])
    relaxed = WorkersConfig(hosts=["here"], yield_to_other_users=False)
    assert worker.run_as_worker(lambda: True, relaxed) == worker.EXIT_OK


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_stop_signal_unwinds_cleanly(tmp_path, monkeypatch, stop_signal):
    monkeypatch.setattr(worker, "other_users", lambda: [])
    job, held = blocked_job(tmp_path)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)  # as if started with `&`
    try:
        threading.Timer(0.5, os.kill, args=(os.getpid(), stop_signal)).start()
        relaxed = WorkersConfig(hosts=["here"], yield_to_other_users=False)
        assert worker.run_as_worker(job, relaxed) == worker.EXIT_STOPPED
    finally:
        signal.signal(signal.SIGINT, previous)
    assert not lock_path(held).exists()
    assert not failed_path(held).exists()


def test_failed_job_gives_exit_code_1(monkeypatch):
    monkeypatch.setattr(worker, "other_users", lambda: [])
    assert worker.run_as_worker(lambda: False, None) == worker.EXIT_FAILURES
