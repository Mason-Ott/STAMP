"""Compare sampling strategies: one config per strategy, each run on its own host.

    python scripts/sampling_sweep.py configs/crc_pipeline.yaml start --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml status --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml stop --n-tiles 1024

Actions: `configs` (only write the configs and show the plan), `start`, `status`,
`stop`. Run it from the STAMP folder, with the environment active. `start` first
checks that every host accepts a passwordless ssh login and starts nothing if one
does not (it says why). Hosts are contacted in parallel.

Strategy i of the sweep goes to host i of `pipeline.workers.hosts` (wrapping
around if there are more strategies than hosts; then a host runs two at once).
Each config is a copy of the base config with the strategy and `n_tiles` set, no
encoder, visualisations on, and `workers.hosts` set to its one host. They are
written to `--out-dir` (default `configs/sweeps/sampling_<n_tiles>`, which must be
on storage every host sees) and started with `stamp workers start`, so logs,
progress and stopping work as for any run (`<output_dir>/logs/<host>.log`).

All strategies share the base config's `output_dir`: tiling and combining are done
once between them (the hosts split the slides), and each strategy then writes its
own folder, e.g. `overviews/sampled/<patient folder>/<strategy>-<n_tiles>/`.
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from stamp.distributed.config import WorkersConfig
from stamp.distributed.remote import SSH_OPTIONS, RemoteRun, host_states, start, stop

STEPS = {
    "first_n": "tile,combine,sample",
    "random": "tile,combine,sample",
    "tissue": "tile,combine,sample",
    "spatial": "tile,combine,sample",
    "entropy": "tile,combine,sample",
    "texture": "tile,combine,sample",
    "yottixel": "tile,combine,sample",
    # clusters tile features, so every tile goes through the extractor first
    "conch_kmeans": "tile,extract_slides,combine,sample",
}
"""Strategy -> pipeline steps to run for it (comma separated)."""


@dataclass(frozen=True)
class Variant:
    strategy: str
    host: str
    steps: str
    config: dict


def plan_variants(
    base: dict, strategies: list[str], n_tiles: int, hosts: list[str] | None = None
) -> list[Variant]:
    """One variant per strategy, assigned to hosts in turn."""
    workers = base.get("pipeline", {}).get("workers")
    if not workers or not (hosts or workers.get("hosts")):
        raise ValueError("the config needs pipeline.workers.hosts (or pass --hosts)")
    hosts = hosts or workers["hosts"]

    variants = []
    for i, strategy in enumerate(strategies):
        config = yaml.safe_load(yaml.safe_dump(base))  # deep copy
        config["preprocessing"]["save_visualisations"] = True
        pipeline = config["pipeline"]
        for key in ("encoder", "steps", "slide_feat_dir"):
            pipeline.pop(key, None)
        pipeline["order"] = (
            "extract_then_sample"
            if strategy == "conch_kmeans"
            else "sample_then_extract"
        )
        pipeline["sampling"] = {
            **pipeline.get("sampling", {}),
            "strategy": strategy,
            "n_tiles": n_tiles,
        }
        host = hosts[i % len(hosts)]
        pipeline["workers"] = {**workers, "hosts": [host]}
        variants.append(Variant(strategy, host, STEPS[strategy], config))
    return variants


def write_configs(variants: list[Variant], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in variants:
        path = out_dir / f"sampling_{variant.strategy}.yaml"
        path.write_text(yaml.safe_dump(variant.config, sort_keys=False))
        paths[variant.strategy] = path
    return paths


def remote_run(variant: Variant, config_file: Path) -> RemoteRun:
    workers = variant.config["pipeline"]["workers"]
    return RemoteRun(
        config_file=config_file,
        output_dir=Path(variant.config["preprocessing"]["output_dir"]).resolve(),
        workers=WorkersConfig(**workers),
        project_dir=workers.get("project_dir") or str(Path.cwd()),
    )


def check_ssh(host: str) -> str | None:
    """None if `ssh host` works without a password, else why not."""
    try:
        result = subprocess.run(
            ["ssh", *SSH_OPTIONS, host, "true"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "timed out"
    if result.returncode == 0:
        return None
    lines = result.stderr.strip().splitlines()
    return lines[-1] if lines else f"ssh exit code {result.returncode}"


def check_hosts(hosts: list[str]) -> dict[str, str | None]:
    with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as pool:
        return dict(zip(hosts, pool.map(check_ssh, hosts), strict=True))


def parallel[T](function, items: list) -> list[T]:
    with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
        return list(pool.map(function, items))


def show_progress(variants: list[Variant], paths: dict[str, Path]) -> None:
    """Per strategy: how far its steps have got (reads the output files)."""
    # imported late: loads the models' libraries, which takes a while
    from stamp.preprocessing.config import PreprocessingConfig
    from stamp.preprocessing.pipeline.config import PipelineConfig, StepName
    from stamp.preprocessing.pipeline.run import pipeline_progress

    print("\nprogress (done/total):")
    for variant in variants:
        pre = PreprocessingConfig.model_validate(variant.config["preprocessing"])
        cfg = PipelineConfig.model_validate(variant.config["pipeline"])
        progress = pipeline_progress(pre, cfg)
        cells = [
            f"{step} {progress[StepName(step)].done}/{progress[StepName(step)].total}"
            for step in variant.steps.split(",")
        ]
        print(f"  {variant.strategy:<13} " + "   ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base_config", type=Path)
    parser.add_argument(
        "action", choices=["configs", "start", "status", "stop"], default="configs"
    )
    parser.add_argument("--n-tiles", type=int, default=1024)
    parser.add_argument("--strategies", nargs="+", default=list(STEPS), choices=STEPS)
    parser.add_argument("--hosts", nargs="+", help="default: pipeline.workers.hosts")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="stop: kill workers immediately"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="status: skip the progress table"
    )
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    variants = plan_variants(base, args.strategies, args.n_tiles, args.hosts)
    out_dir = (
        args.out_dir or Path("configs/sweeps") / f"sampling_{args.n_tiles}"
    ).resolve()
    paths = write_configs(variants, out_dir)
    runs = {v.strategy: remote_run(v, paths[v.strategy]) for v in variants}

    print(f"configs in {out_dir}")
    for variant in variants:
        print(f"  {variant.strategy:<13} {variant.host:<12} steps: {variant.steps}")
    if args.action == "configs":
        return

    hosts = [v.host for v in variants]
    reasons = check_hosts(sorted(set(hosts)))
    broken = {host: why for host, why in reasons.items() if why}
    if broken:
        print("\nhosts that do not accept a passwordless ssh login:")
        for host, why in broken.items():
            print(f"  {host}: {why}")
        print(
            "\nFor 'Permission denied', run once per host: "
            "ssh-copy-id -i ~/.ssh/id_ed25519.pub <host>\n"
            "Replace hosts that do not exist (Could not resolve) in the config, or "
            "choose hosts with --hosts."
        )

    match args.action:
        case "start":
            if broken:
                print("\nnothing started.")
                raise SystemExit(1)
            outcomes = parallel(
                lambda v: start(runs[v.strategy], [v.host], ["--steps", v.steps])[
                    v.host
                ],
                variants,
            )
            print()
            for variant, outcome in zip(variants, outcomes, strict=True):
                print(f"{variant.strategy:<13} {variant.host:<12} {outcome}")
            print(f"\nlogs: {next(iter(runs.values())).log_file('<host>')}")
            print("check progress with the `status` action")
        case "stop":
            outcomes = parallel(
                lambda v: stop(runs[v.strategy], [v.host], force=args.force)[v.host],
                [v for v in variants if v.host not in broken],
            )
            for variant, outcome in zip(
                [v for v in variants if v.host not in broken], outcomes, strict=True
            ):
                print(f"{variant.strategy:<13} {variant.host:<12} {outcome}")
        case "status":
            reachable = [v for v in variants if v.host not in broken]
            states = parallel(
                lambda v: host_states(runs[v.strategy], [v.host])[v.host].value,
                reachable,
            )
            state_of = {v.strategy: s for v, s in zip(reachable, states, strict=True)}
            print()
            for variant in variants:
                state = state_of.get(variant.strategy, "unreachable")
                print(f"{variant.strategy:<13} {variant.host:<12} worker {state}")
            if not args.no_progress:
                show_progress(variants, paths)


if __name__ == "__main__":
    main()
