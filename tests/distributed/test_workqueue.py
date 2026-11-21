import os
import threading
import time
from pathlib import Path

import pytest

from stamp.distributed.workqueue import (
    InputState,
    Step,
    claim,
    failed_path,
    lock_path,
    run_step,
    wait_for,
)
from stamp.preprocessing.pipeline.config import CoordinationConfig

FAST = CoordinationConfig(lock_stale_after_s=2.0, poll_interval_s=0.05)


def writer(item: str, inputs: list[Path], output: Path) -> None:
    output.write_text(f"{item} from {[p.name for p in inputs]}")


def step(tmp_path: Path, items: list[str], **kwargs) -> Step[str]:
    return Step(
        name="test",
        items=items,
        output_of=lambda item: tmp_path / f"{item}.out",
        inputs_of=kwargs.pop("inputs_of", lambda item: []),
        run=kwargs.pop("run", writer),
        **kwargs,
    )


def test_runs_every_item_once(tmp_path):
    report = run_step(step(tmp_path, ["a", "b"]), FAST)
    assert report.done == 2
    assert run_step(step(tmp_path, ["a", "b"]), FAST).already_done == 2


def test_failure_leaves_marker_and_is_skipped_next_time(tmp_path):
    def broken(item, inputs, output):
        raise RuntimeError("boom")

    report = run_step(step(tmp_path, ["a"], run=broken), FAST)
    assert report.failed == 1 and report.failures == [tmp_path / "a.out"]
    assert "boom" in failed_path(tmp_path / "a.out").read_text()

    assert run_step(step(tmp_path, ["a"]), FAST).failed_earlier == 1
    retry = CoordinationConfig(**{**FAST.model_dump(), "retry_failed": True})
    assert run_step(step(tmp_path, ["a"]), retry).done == 1
    assert not failed_path(tmp_path / "a.out").exists()


def test_item_locked_elsewhere_is_skipped(tmp_path):
    lock_path(tmp_path / "a.out").write_text("other machine")
    report = run_step(step(tmp_path, ["a", "b"]), FAST)
    assert report.busy_elsewhere == 1 and report.done == 1


def test_stale_lock_is_taken_over(tmp_path):
    lock = lock_path(tmp_path / "a.out")
    lock.write_text("dead machine")
    old = time.time() - 10
    os.utime(lock, (old, old))
    assert run_step(step(tmp_path, ["a"]), FAST).done == 1
    assert not lock.exists()


def test_claim_is_exclusive_and_released(tmp_path):
    target = tmp_path / "x"
    with claim(target, 10.0) as first, claim(target, 10.0) as second:
        assert first and not second
    with claim(target, 10.0) as again:
        assert again


def test_missing_input_skips_the_item_without_marking_it_failed(tmp_path):
    needs_input = step(tmp_path, ["a"], inputs_of=lambda item: [tmp_path / "x.in"])
    report = run_step(needs_input, FAST)
    assert report.missing_inputs == 1 and report.failed == 0 and not report.ok
    assert not failed_path(tmp_path / "a.out").exists()

    # Once the input exists (e.g. the earlier step has run), the item just runs.
    (tmp_path / "x.in").write_text("")
    assert run_step(needs_input, FAST).done == 1


def test_partial_inputs_when_allowed(tmp_path):
    (tmp_path / "1.in").write_text("")
    inputs = [tmp_path / "1.in", tmp_path / "2.in"]
    report = run_step(
        step(tmp_path, ["a"], inputs_of=lambda item: inputs, allow_missing_inputs=True),
        FAST,
    )
    assert report.done == 1
    assert (tmp_path / "a.out").read_text() == "a from ['1.in']"


def test_waits_for_input_being_produced_elsewhere(tmp_path):
    produced = tmp_path / "in.h5"
    lock_path(produced).write_text("other machine")

    def finish_later():
        time.sleep(0.3)
        produced.write_text("")
        lock_path(produced).unlink()

    threading.Thread(target=finish_later).start()
    started = time.time()
    state = wait_for(produced, stale_after_s=5.0, poll_interval_s=0.05)
    assert state is InputState.READY and time.time() - started >= 0.25


@pytest.mark.parametrize(
    "setup, expected",
    [
        (lambda p: None, InputState.MISSING),
        (lambda p: failed_path(p).write_text(""), InputState.FAILED),
        (lambda p: p.write_text(""), InputState.READY),
    ],
)
def test_wait_for_returns_immediately_when_nobody_is_working(tmp_path, setup, expected):
    path = tmp_path / "in.h5"
    setup(path)
    assert wait_for(path, stale_after_s=5.0, poll_interval_s=10.0) is expected


def test_item_is_done_when_a_later_output_exists(tmp_path):
    (tmp_path / "later.out").write_text("")
    report = run_step(
        step(tmp_path, ["a"], later_outputs_of=lambda item: [tmp_path / "later.out"]),
        FAST,
    )
    assert report.already_done == 1 and not (tmp_path / "a.out").exists()


def test_inputs_are_deleted_only_after_the_output_is_written(tmp_path):
    (tmp_path / "a.in").write_text("")

    def broken(item, inputs, output):
        raise RuntimeError("boom")

    consuming = {"inputs_of": lambda item: [tmp_path / "a.in"]}
    run_step(
        step(tmp_path, ["a"], run=broken, delete_inputs_when_done=True, **consuming),
        FAST,
    )
    assert (tmp_path / "a.in").exists(), "kept for recovery after a failure"

    retry = CoordinationConfig(**{**FAST.model_dump(), "retry_failed": True})
    run_step(step(tmp_path, ["a"], delete_inputs_when_done=True, **consuming), retry)
    assert (tmp_path / "a.out").exists() and not (tmp_path / "a.in").exists()


def test_items_whose_inputs_exist_run_before_items_waiting_for_inputs(tmp_path):
    order = []

    def recording(item, inputs, output):
        order.append(item)
        writer(item, inputs, output)

    ready = {"r1", "r2", "r3"}
    for name in ready:
        (tmp_path / f"{name}.in").write_text("")
    # w1..w3 wait for an input that is being produced elsewhere (locked, so the
    # run waits for it; it never appears, so they end up skipped)
    items = ["w1", "r1", "w2", "r2", "w3", "r3"]
    waiting_step = step(
        tmp_path,
        items,
        inputs_of=lambda item: [tmp_path / f"{item}.in"],
        run=recording,
    )
    run_step(waiting_step, FAST)
    assert set(order[:3]) == ready, "everything that could run went first"
    assert set(order) == ready, "the others have no inputs and were skipped"


def test_ensure_dir_survives_other_machines_creating_the_same_folders(
    tmp_path, monkeypatch
):
    from stamp.distributed import fs

    real_mkdir = Path.mkdir
    failures = {"left": 3}

    def racing_mkdir(self, *args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            real_mkdir(self, *args, **kwargs)  # the other machine got there first
            raise FileExistsError(self)  # ...and our view of it is stale
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    monkeypatch.setattr(fs.time, "sleep", lambda s: None)
    target = tmp_path / "a" / "b" / "c"
    fs.ensure_dir(target)
    assert target.is_dir() and failures["left"] == 0


def test_ensure_dir_gives_up_eventually(tmp_path, monkeypatch):
    from stamp.distributed import fs

    def always_fails(self, *args, **kwargs):
        raise FileExistsError(self)

    monkeypatch.setattr(Path, "mkdir", always_fails)
    monkeypatch.setattr(fs.time, "sleep", lambda s: None)
    with pytest.raises(FileExistsError):
        fs.ensure_dir(tmp_path / "x", attempts=3)


def test_many_threads_creating_the_same_folders_all_succeed(tmp_path):
    from stamp.distributed.fs import ensure_dir

    errors = []

    def create(i):
        try:
            ensure_dir(tmp_path / "shared" / "deep" / "er" / f"{i % 3}")
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(24)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
