"""Settings for running work on several machines (see workqueue.py and remote.py)."""

from pydantic import BaseModel, ConfigDict, Field


class CoordinationConfig(BaseModel):
    """How workers on several machines share a step through lock files."""

    model_config = ConfigDict(extra="forbid")

    lock_stale_after_s: float = Field(default=900.0, gt=0)
    """A lock not refreshed for this long is treated as abandoned (dead machine)."""

    poll_interval_s: float = Field(default=30.0, gt=0)
    """How often to check for inputs that another machine is still producing."""

    retry_failed: bool = False
    """Retry items that have a `.failed` marker from an earlier run."""


class WorkersConfig(BaseModel):
    """Machines for `stamp workers start|status|stop` (run from your own machine).

    Every host needs the same filesystem view as this one: the config file and
    every input and output folder at the same paths.
    """

    model_config = ConfigDict(extra="forbid")

    hosts: list[str]
    """SSH host names, as you would type them after `ssh`."""

    project_dir: str | None = None
    """Folder to `cd` into on each host before running. Default: the folder
    `stamp workers start` is run from."""

    activate: str | None = None
    """Shell command that makes `python -m stamp` work on the hosts, e.g.
    `source ~/stamp/.venv/bin/activate`. Run in a login shell."""

    yield_to_other_users: bool = True
    """Stop a worker cleanly when someone other than you logs into its machine."""

    presence_check_s: float = Field(default=60.0, gt=0)
    """How often a worker checks who is logged in."""
