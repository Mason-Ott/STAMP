"""`stamp workers start|status|stop`: run `stamp worker` on other machines over SSH.

Each host gets one worker inside a detached `screen` session, so it survives the
SSH connection closing. The workers share the work through the pipeline's lock
files, so there is no queue or server: starting a worker on another host just
adds capacity, and stopping one leaves its unfinished item to the others.

Needs passwordless SSH (keys) to every host, `screen` on every host, and the same
paths everywhere (see WorkersConfig).
"""

import hashlib
import logging
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

from stamp.distributed.config import WorkersConfig

_logger = logging.getLogger("stamp")

T = TypeVar("T")

SSH_OPTIONS = (
    "-o", "BatchMode=yes",  # fail instead of prompting for a password
    "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
)  # fmt: skip

SSH_UNREACHABLE = 255
"""ssh's own exit code when it can't connect or log in."""


class HostState(Enum):
    RUNNING = "running"
    STOPPED = "not running"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class RemoteRun:
    """Everything needed to start, inspect or stop the workers of one run."""

    config_file: Path
    """Absolute path; must be the same on every host."""
    output_dir: Path
    workers: WorkersConfig
    project_dir: str

    @property
    def session(self) -> str:
        """screen session name, also used to find the worker process.

        Includes a hash of the config file's path, so two runs never share a name
        (stop must never signal another run's worker)."""
        readable = re.sub(r"[^A-Za-z0-9_-]+", "_", self.output_dir.name)
        digest = hashlib.sha256(str(self.config_file).encode()).hexdigest()[:8]
        return f"stamp_{readable}_{digest}"

    def log_file(self, host: str) -> Path:
        return self.output_dir / "logs" / f"{host}.log"

    def hosts(self, only: Sequence[str] | None) -> list[str]:
        if not only:
            return list(self.workers.hosts)
        if unknown := sorted(set(only) - set(self.workers.hosts)):
            raise ValueError(f"hosts {unknown} are not in pipeline.workers.hosts")
        return list(only)


def start(
    run: RemoteRun, hosts: Sequence[str], worker_args: Sequence[str] = ()
) -> dict[str, str]:
    """Start a worker on each host that doesn't have one yet. Host -> outcome.

    `worker_args` are added to the `stamp worker` command line (e.g. --steps).
    """

    def start_on(host: str) -> str:
        result = _ssh(host, _start_command(run, host, worker_args))
        if result.returncode == 0:
            return result.stdout.strip() or "started"
        return f"failed: {_error(result)}"

    return _on_hosts(hosts, start_on)


def stop(run: RemoteRun, hosts: Sequence[str], *, force: bool) -> dict[str, str]:
    """Ask each host's worker to stop (SIGTERM), or kill its session if `force`.

    A normal stop lets the worker release its locks. A forced stop doesn't, so
    its unfinished item waits until the lock goes stale.
    """
    if force:
        command = f"screen -S {run.session} -X quit"
    else:
        # -o: only the worker itself, not its DataLoader child processes
        pattern = f"worker --session {run.session}( |$)"
        command = f'pkill -TERM -o -u "$(id -u)" -f -- {shlex.quote(pattern)}'

    def stop_on(host: str) -> str:
        result = _ssh(host, command)
        if result.returncode == SSH_UNREACHABLE:
            return f"failed: {_error(result)}"
        return "stopping" if result.returncode == 0 else "no worker running"

    return _on_hosts(hosts, stop_on)


def host_states(run: RemoteRun, hosts: Sequence[str]) -> dict[str, HostState]:
    def state_of(host: str) -> HostState:
        result = _ssh(host, f"if {_session_exists(run)}; then echo yes; fi")
        if result.returncode == SSH_UNREACHABLE:
            return HostState.UNREACHABLE
        return HostState.RUNNING if "yes" in result.stdout else HostState.STOPPED

    return _on_hosts(hosts, state_of)


def _start_command(run: RemoteRun, host: str, worker_args: Sequence[str]) -> str:
    """Shell command, run on `host`, that starts the worker unless one is running."""
    worker = [
        "python", "-m", "stamp", "-c", str(run.config_file),
        "worker", "--session", run.session, *worker_args,
    ]  # fmt: skip

    log = shlex.quote(str(run.log_file(host)))
    setup = [f"cd {shlex.quote(run.project_dir)}"]
    if run.workers.activate:
        setup.append(run.workers.activate)  # a shell snippet, used as written
    setup += [
        f"mkdir -p {shlex.quote(str(run.log_file(host).parent))}",
        f'echo "=== $(date) worker starting on $(hostname)" >> {log}',
        # TQDM_DISABLE keeps progress bars out of the log file
        f"TQDM_DISABLE=1 exec {shlex.join(worker)} >> {log} 2>&1",
    ]
    in_screen = f"screen -dmS {run.session} bash -lc {shlex.quote(' && '.join(setup))}"
    return (
        f"if {_session_exists(run)}; then echo 'already running'; "
        f"else {in_screen} && echo started; fi"
    )


def _session_exists(run: RemoteRun) -> str:
    """Shell test that succeeds if this run's screen session exists."""
    return f"screen -ls 2>/dev/null | grep -qE '[0-9]+\\.{run.session}[[:space:]]'"


def _ssh(host: str, command: str, timeout_s: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["ssh", *SSH_OPTIONS, host, command],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], SSH_UNREACHABLE, "", "timed out")


def _on_hosts(hosts: Sequence[str], action: Callable[[str], T]) -> dict[str, T]:
    """Run `action` for every host in parallel; results in the order of `hosts`."""
    with ThreadPoolExecutor(max_workers=max(1, min(16, len(hosts)))) as pool:
        results = list(pool.map(action, hosts))
    return dict(zip(hosts, results, strict=True))


def _error(result: subprocess.CompletedProcess) -> str:
    lines = (result.stderr or result.stdout).strip().splitlines()
    return lines[-1] if lines else f"exit code {result.returncode}"
