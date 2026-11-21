"""Command-line glue for `stamp workers start|status|stop`."""

from pathlib import Path
from typing import Literal

from stamp.distributed.progress import format_progress
from stamp.distributed.remote import RemoteRun, host_states, start, stop
from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.pipeline.config import PipelineConfig, StepName
from stamp.preprocessing.pipeline.plan import layout_for
from stamp.preprocessing.pipeline.run import pipeline_progress

WorkersAction = Literal["start", "status", "stop"]


def workers_command(
    pre: PreprocessingConfig,
    cfg: PipelineConfig,
    *,
    config_file: Path,
    action: WorkersAction,
    hosts: list[str] | None,
    steps: list[StepName] | None,
    force: bool,
) -> None:
    if cfg.workers is None:
        raise ValueError(
            "`stamp workers` needs a pipeline.workers section in the config"
        )
    run = RemoteRun(
        config_file=config_file.resolve(),
        output_dir=layout_for(pre, cfg).run_dir.resolve(),
        workers=cfg.workers,
        project_dir=cfg.workers.project_dir or str(Path.cwd()),
    )
    selected = run.hosts(hosts)

    match action:
        case "start":
            worker_args = ["--steps", ",".join(map(str, steps))] if steps else []
            _print_hosts(start(run, selected, worker_args))
            print(f"\nlogs: {run.log_file('<host>')}")
            print(f"progress: stamp -c {config_file} workers status")
        case "stop":
            _print_hosts(stop(run, selected, force=force))
        case "status":
            print(format_progress(pipeline_progress(pre, cfg).values()))
            print()
            _print_hosts({h: s.value for h, s in host_states(run, selected).items()})


def _print_hosts(outcomes: dict[str, str]) -> None:
    width = max(len(host) for host in outcomes)
    for host, outcome in outcomes.items():
        print(f"{host:<{width}}  {outcome}")
