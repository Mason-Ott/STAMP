"""`stamp workers start|status|stop` running pipeline workers, with stand-ins for
ssh and screen.

The stand-in `ssh` runs the command on this machine (as if every host were this
one); hosts named `unreachable*` fail the way ssh does. The stand-in `screen`
supports only what remote.py uses: -dmS, -ls and -S <name> -X quit.
"""

import contextlib
import os
import re
import signal
import stat
import time
from pathlib import Path

import pytest
import yaml

from stamp.distributed.remote import HostState, RemoteRun, host_states, start, stop
from stamp.distributed.workqueue import failed_path, lock_path
from stamp.preprocessing.data.bags import read_bag
from stamp.preprocessing.pipeline.config import StepName

from .conftest import SLIDES, TILE_SIZE_PX, TILE_SIZE_UM

FAKE_SSH = """#!/usr/bin/env bash
while [[ "$1" == -o ]]; do shift 2; done
host="$1"; shift
if [[ "$host" == unreachable* ]]; then
  echo "ssh: connect to host $host port 22: No route to host" >&2; exit 255
fi
export FAKE_HOST="$host"
exec bash -c "$*"
"""

FAKE_SCREEN = """#!/usr/bin/env bash
dir="$FAKE_SCREEN_ROOT/${FAKE_HOST:-local}"; mkdir -p "$dir"
case "$1" in
  -dmS) name="$2"; shift 2
        setsid "$@" </dev/null >/dev/null 2>&1 &
        echo $! > "$dir/$name.pid" ;;
  -ls)  found=0
        for f in "$dir"/*.pid; do
          [ -e "$f" ] || continue; pid=$(cat "$f")
          if kill -0 "$pid" 2>/dev/null; then
            printf '\\t%s.%s\\t(Detached)\\n' "$pid" "$(basename "$f" .pid)"; found=1
          fi
        done
        [ "$found" = 1 ] || { echo "No Sockets found"; exit 1; } ;;
  -S)   [ "$3 $4" = "-X quit" ] && kill -HUP -- "-$(cat "$dir/$2.pid")" ;;
esac
"""


@pytest.fixture
def fake_cluster(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in {"ssh": FAKE_SSH, "screen": FAKE_SCREEN}.items():
        path = bin_dir / name
        path.write_text(script)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SCREEN_ROOT", str(tmp_path / "screens"))
    yield
    # Never leave workers behind, whatever the test did.
    for pid_file in (tmp_path / "screens").rglob("*.pid"):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(int(pid_file.read_text()), signal.SIGKILL)


@pytest.fixture
def remote_run(dataset, tmp_path) -> RemoteRun:
    from stamp.preprocessing.pipeline.config import WorkersConfig

    project_dir = tmp_path / "my project"  # a space, to check the quoting
    project_dir.mkdir()
    output_dir = tmp_path / "out"
    config_file = tmp_path / "config.yaml"
    workers = {
        "hosts": ["hostA", "hostB", "unreachable1"],
        "project_dir": str(project_dir),
        "activate": f"export PYTHONPATH={os.environ.get('PYTHONPATH', '')}",
        "yield_to_other_users": False,  # the test machine may have other sessions
    }
    config_file.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {
                    "output_dir": str(output_dir),
                    "wsi_dir": str(dataset.wsi_dir),
                    "cache_dir": str(tmp_path / "cache"),
                    "extractor": "conch1_5",
                    "tile_size_um": TILE_SIZE_UM,
                    "tile_size_px": TILE_SIZE_PX,
                    "device": "cpu",
                    "max_workers": 2,
                },
                "pipeline": {
                    "experiment": "e",
                    "clinical_table": str(dataset.clinical_table),
                    "sampling": {"n_tiles": 50},
                    "coordination": {"poll_interval_s": 0.2},
                    "workers": workers,
                },
            }
        )
    )
    return RemoteRun(
        config_file=config_file,
        output_dir=output_dir,
        workers=WorkersConfig(**workers),
        project_dir=str(project_dir),
    )


NO_MODEL_STEPS = ["--steps", "tile,combine,sample"]
COMBINE_ONLY = ["--steps", "combine"]


def wait_until_stopped(
    run: RemoteRun, hosts: list[str], timeout_s: float = 300
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        states = host_states(run, hosts)
        if all(state is HostState.STOPPED for state in states.values()):
            return
        time.sleep(0.5)
    raise TimeoutError(f"workers still running: {states}")


def test_workers_on_two_hosts_share_the_run(fake_cluster, remote_run):
    outcomes = start(remote_run, ["hostA", "hostB"], NO_MODEL_STEPS)
    assert outcomes == {"hostA": "started", "hostB": "started"}
    wait_until_stopped(remote_run, ["hostA", "hostB"])

    logs = {h: remote_run.log_file(h).read_text() for h in ("hostA", "hostB")}
    for host, log in logs.items():
        assert "finished" in log, f"{host} log:\n{log[-2000:]}"
    tiled = sum(
        int(n)
        for log in logs.values()
        for n in re.findall(r"\ttile: .*\bdone=(\d+)", log)
    )
    assert tiled == len(SLIDES), "every slide tiled by exactly one host"

    out = remote_run.output_dir
    assert not list(out.rglob("*.lock")) and not list(out.rglob("*.failed"))
    sampled = sorted(layout_of(remote_run).stage(StepName.SAMPLE).folder.glob("*.h5"))
    assert [p.stem for p in sampled] == ["H1_P1", "H1_P2", "H2_P1"]
    assert all(read_bag(p).sampling is not None for p in sampled)


def test_unreachable_host_is_reported(fake_cluster, remote_run):
    assert start(remote_run, ["unreachable1"], NO_MODEL_STEPS)[
        "unreachable1"
    ].startswith("failed: ssh: connect to host unreachable1")
    assert host_states(remote_run, ["unreachable1"]) == {
        "unreachable1": HostState.UNREACHABLE
    }


def layout_of(run: RemoteRun):
    from stamp.preprocessing.config import PreprocessingConfig
    from stamp.preprocessing.pipeline.config import PipelineConfig
    from stamp.preprocessing.pipeline.plan import layout_for

    settings = yaml.safe_load(run.config_file.read_text())
    return layout_for(
        PreprocessingConfig(**settings["preprocessing"]),
        PipelineConfig(**settings["pipeline"]),
    )


def blocked_run(remote_run: RemoteRun) -> Path:
    """Make `combine` wait forever on one slide, as if another host were tiling it.

    Returns the output the worker will hold a lock on while it waits.
    """
    layout = layout_of(remote_run)
    tiles = layout.stage(StepName.TILE).folder
    tiles.mkdir(parents=True)
    lock_path(tiles / "S1.h5").write_text("elsewhere pid=1")
    return layout.stage(StepName.COMBINE).folder / "H1_P1.h5"


def test_start_twice_and_stop_cleanly(fake_cluster, remote_run):
    held = blocked_run(remote_run)
    assert start(remote_run, ["hostA"], COMBINE_ONLY) == {"hostA": "started"}
    assert start(remote_run, ["hostA"], COMBINE_ONLY) == {"hostA": "already running"}
    _wait_for(lambda: lock_path(held).exists(), "worker to claim H1_P1")

    assert stop(remote_run, ["hostA"], force=False) == {"hostA": "stopping"}
    wait_until_stopped(remote_run, ["hostA"])

    log = remote_run.log_file("hostA").read_text()
    assert "stopped on request" in log, log[-2000:]
    assert not lock_path(held).exists(), "lock should be released on a clean stop"
    assert not failed_path(held).exists(), "a stop is not a failure"
    assert stop(remote_run, ["hostA"], force=False) == {"hostA": "no worker running"}


def test_force_stop_ends_the_session_and_the_worker_unwinds(fake_cluster, remote_run):
    held = blocked_run(remote_run)
    start(remote_run, ["hostA"], COMBINE_ONLY)
    _wait_for(lambda: lock_path(held).exists(), "worker to claim H1_P1")

    stop(remote_run, ["hostA"], force=True)
    wait_until_stopped(remote_run, ["hostA"], timeout_s=30)
    # The session's hangup is handled like a stop request: the worker logs it and
    # releases its lock instead of leaving one to go stale.
    _wait_for(lambda: not lock_path(held).exists(), "the lock to be released")
    assert "stopped on request" in remote_run.log_file("hostA").read_text()


def _wait_for(condition, what: str, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while not condition():
        if time.time() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(0.2)


def test_session_names_differ_between_runs(remote_run):
    other = RemoteRun(
        config_file=remote_run.config_file.with_name("other.yaml"),
        output_dir=remote_run.output_dir,
        workers=remote_run.workers,
        project_dir=remote_run.project_dir,
    )
    assert remote_run.session.startswith("stamp_out_")
    assert remote_run.session != other.session
