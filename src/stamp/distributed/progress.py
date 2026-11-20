"""Progress of a step, read from its output files (no running process needed)."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from stamp.distributed.workqueue import Step, failed_path, lock_holder


@dataclass
class StepProgress:
    name: str
    total: int
    done: int = 0
    failed: int = 0
    running_on: Counter[str] = field(default_factory=Counter)
    """Host -> items it is producing right now."""

    @property
    def running(self) -> int:
        return sum(self.running_on.values())

    @property
    def waiting(self) -> int:
        return self.total - self.done - self.failed - self.running


def step_progress(step: Step, stale_after_s: float) -> StepProgress:
    progress = StepProgress(step.name, total=len(step.items))
    for item in step.items:
        output = step.output_of(item)
        if step.is_done(item):
            progress.done += 1
        elif host := lock_holder(output, stale_after_s):
            progress.running_on[host] += 1
        elif failed_path(output).exists():
            progress.failed += 1
    return progress


def format_progress(steps: Iterable[StepProgress]) -> str:
    """A table with one row per step."""
    lines = [
        f"{'STEP':<16}{'DONE':>12}{'RUNNING':>9}{'FAILED':>8}{'WAITING':>9}  HOSTS"
    ]
    for step in steps:
        hosts = ", ".join(f"{h} ({n})" for h, n in sorted(step.running_on.items()))
        lines.append(
            f"{step.name:<16}{f'{step.done}/{step.total}':>12}{step.running:>9}"
            f"{step.failed:>8}{step.waiting:>9}  {hosts}"
        )
    return "\n".join(lines)
