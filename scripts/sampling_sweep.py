"""Compare sampling strategies: one config per strategy, each run on its own host.

    python scripts/sampling_sweep.py configs/crc_pipeline.yaml ssh-setup
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml start --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml status --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml stop --n-tiles 1024

Run it from the STAMP folder, with the environment active.

Actions
  ssh-setup  for every host in the pool that does not accept your key yet, run
             `ssh-copy-id` (asks for your password once per host), then re-check.
  configs    only write the configs and show the plan (no ssh).
  start      check the hosts, write the configs and start one worker per strategy.
  status     worker state and progress of every strategy.
  stop       stop the workers (--force kills them).

`pipeline.workers.hosts` is a **pool**. `start` checks every host in parallel,
skips the ones that do not accept a passwordless login (and says why), and gives
the strategies, in order, to the working hosts. If starting on a host fails
anyway, the strategy moves to the next unused host of the pool. Only if the pool
has fewer working hosts than strategies does a host run two at once. Put more
hosts in the pool than strategies to have spares.

Each config is a copy of the base config with the strategy and `n_tiles` set, no
encoder, visualisations on, and `workers.hosts` set to its one host. They are
written to `--out-dir` (default `configs/sweeps/sampling_<n_tiles>`, which must be
on storage every host sees) and started with `stamp workers start`, so logs and
progress work as for any run (`<output_dir>/logs/<host>.log`). `status` and `stop`
use the hosts recorded in those files.

All strategies share the base config's `output_dir`: tiling and combining are done
once between them (the hosts split the slides), and each strategy then writes its
own folder, e.g. `overviews/sampled/<patient folder>/<strategy>-<n_tiles>/`.
"""

import argparse
import dataclasses
import subprocess
from collections.abc import Callable
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


# --- planning (no ssh) ---------------------------------------------------------


def plan_variants(
    base: dict, strategies: list[str], n_tiles: int, hosts: list[str] | None = None
) -> list[Variant]:
    """One variant per strategy, assigned to `hosts` (default: the config's pool)
    in turn."""
    workers = base.get("pipeline", {}).get("workers")
    if not workers or not (hosts or workers.get("hosts")):
        raise ValueError("the config needs pipeline.workers.hosts (or pass --hosts)")
    hosts = hosts or workers["hosts"]

    variants = []
    for strategy in strategies:
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
        variants.append(Variant(strategy, "", STEPS[strategy], config))
    return [with_host(v, hosts[i % len(hosts)]) for i, v in enumerate(variants)]


def with_host(variant: Variant, host: str) -> Variant:
    config = yaml.safe_load(yaml.safe_dump(variant.config))
    workers = config["pipeline"]["workers"]
    config["pipeline"]["workers"] = {**workers, "hosts": [host]}
    return dataclasses.replace(variant, host=host, config=config)


def write_configs(variants: list[Variant], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in variants:
        paths[variant.strategy] = config_path(out_dir, variant.strategy)
        paths[variant.strategy].write_text(
            yaml.safe_dump(variant.config, sort_keys=False)
        )
    return paths


def config_path(out_dir: Path, strategy: str) -> Path:
    return out_dir / f"sampling_{strategy}.yaml"


def load_variants(out_dir: Path, strategies: list[str]) -> list[Variant]:
    """The variants as written by `start`/`configs` (with the hosts they got)."""
    variants = []
    for strategy in strategies:
        path = config_path(out_dir, strategy)
        if not path.exists():
            raise SystemExit(f"{path} does not exist: run `start` first")
        config = yaml.safe_load(path.read_text())
        host = config["pipeline"]["workers"]["hosts"][0]
        variants.append(Variant(strategy, host, STEPS[strategy], config))
    return variants


def working_hosts(pool: list[str], problems: dict[str, str | None]) -> list[str]:
    """The pool's hosts that passed the check, in pool order."""
    return [host for host in pool if not problems.get(host)]


def start_all(
    variants: list[Variant],
    spares: list[str],
    starter: Callable[[Variant], str],
) -> list[tuple[Variant, str]]:
    """Start every variant on its host (in parallel). One that fails moves to the
    next spare host, until it starts or there are no spares left. Returns each
    variant (with the host it ended up on) and the last outcome."""
    results = parallel(lambda v: (v, starter(v)), variants)
    spares = list(spares)
    final = []
    for variant, outcome in results:
        while is_failure(outcome) and spares:
            variant = with_host(variant, spares.pop(0))
            outcome = starter(variant)
        final.append((variant, outcome))
    return final


def is_failure(outcome: str) -> bool:
    return outcome.startswith("failed")


# --- ssh -----------------------------------------------------------------------


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
    return dict(zip(hosts, parallel(check_ssh, hosts), strict=True))


def setup_ssh(pool: list[str], key: Path) -> None:
    """ssh-copy-id to every host that does not accept the key yet, one after
    another because each asks for your password."""
    if not key.exists():
        raise SystemExit(
            f"{key} does not exist; create one with: ssh-keygen -t ed25519"
        )
    problems = check_hosts(pool)
    for host in pool:
        why = problems[host]
        if why is None:
            print(f"{host}: already works")
        elif "resolve" in why.lower():
            print(f"{host}: {why} (not a host name here; remove it from the pool)")
        else:
            print(f"\n{host}: {why}\ninstalling {key.name}; enter your password:")
            subprocess.run(
                [
                    "ssh-copy-id",
                    "-i",
                    str(key),
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    host,
                ]
            )
    print("\nafter setup:")
    for host, why in check_hosts(pool).items():
        print(f"  {host}: {'ok' if why is None else why}")


def parallel[T, R](function: Callable[[T], R], items: list[T]) -> list[R]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return list(pool.map(function, items))


# --- output --------------------------------------------------------------------


def show_progress(variants: list[Variant]) -> None:
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


def print_plan(variants: list[Variant], out_dir: Path) -> None:
    print(f"configs in {out_dir}")
    for variant in variants:
        print(f"  {variant.strategy:<13} {variant.host:<12} steps: {variant.steps}")


def explain_problems(problems: dict[str, str | None]) -> None:
    broken = {host: why for host, why in problems.items() if why}
    if not broken:
        return
    print("\nhosts skipped (no passwordless ssh login):")
    for host, why in broken.items():
        print(f"  {host}: {why}")
    print("  fix 'Permission denied' for all of them with the `ssh-setup` action")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base_config", type=Path)
    parser.add_argument(
        "action",
        choices=["ssh-setup", "configs", "start", "status", "stop"],
        default="configs",
    )
    parser.add_argument("--n-tiles", type=int, default=1024)
    parser.add_argument("--strategies", nargs="+", default=list(STEPS), choices=STEPS)
    parser.add_argument("--hosts", nargs="+", help="the pool (default: from config)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--key", type=Path, default=Path("~/.ssh/id_ed25519.pub"))
    parser.add_argument("--force", action="store_true", help="stop: kill immediately")
    parser.add_argument("--no-progress", action="store_true", help="skip progress")
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    pool = args.hosts or base.get("pipeline", {}).get("workers", {}).get("hosts", [])
    if not pool:
        raise SystemExit("the config needs pipeline.workers.hosts (or pass --hosts)")
    out_dir = (
        args.out_dir or Path("configs/sweeps") / f"sampling_{args.n_tiles}"
    ).resolve()

    match args.action:
        case "ssh-setup":
            setup_ssh(pool, args.key.expanduser())

        case "configs":
            variants = plan_variants(base, args.strategies, args.n_tiles, pool)
            write_configs(variants, out_dir)
            print_plan(variants, out_dir)

        case "start":
            problems = check_hosts(pool)
            usable = working_hosts(pool, problems)
            explain_problems(problems)
            if not usable:
                raise SystemExit(
                    "\nno host accepts a passwordless login: nothing started"
                )
            variants = plan_variants(base, args.strategies, args.n_tiles, usable)
            spares = usable[len(variants) :]
            if len(usable) < len(variants):
                print(
                    f"\nonly {len(usable)} working hosts for {len(variants)} "
                    "strategies: some hosts will run two at once"
                )
            paths = write_configs(variants, out_dir)

            def starter(variant: Variant) -> str:
                path = write_configs([variant], out_dir)[variant.strategy]
                return start(
                    remote_run(variant, path),
                    [variant.host],
                    ["--steps", variant.steps],
                )[variant.host]

            results = start_all(variants, spares, starter)
            print()
            for variant, outcome in results:
                print(f"{variant.strategy:<13} {variant.host:<12} {outcome}")
            first = remote_run(results[0][0], paths[results[0][0].strategy])
            print(f"\nlogs: {first.log_file('<host>')}")
            if any(is_failure(outcome) for _, outcome in results):
                raise SystemExit(1)

        case "status" | "stop":
            variants = load_variants(out_dir, args.strategies)
            problems = check_hosts(sorted({v.host for v in variants}))
            explain_problems(problems)
            reachable = [v for v in variants if not problems[v.host]]
            runs = {
                v.strategy: remote_run(v, config_path(out_dir, v.strategy))
                for v in variants
            }
            if args.action == "stop":
                outcomes = parallel(
                    lambda v: stop(runs[v.strategy], [v.host], force=args.force)[
                        v.host
                    ],
                    reachable,
                )
            else:
                outcomes = parallel(
                    lambda v: (
                        "worker "
                        + host_states(runs[v.strategy], [v.host])[v.host].value
                    ),
                    reachable,
                )
            result_of = {
                v.strategy: o for v, o in zip(reachable, outcomes, strict=True)
            }
            print()
            for variant in variants:
                outcome = result_of.get(variant.strategy, "unreachable")
                print(f"{variant.strategy:<13} {variant.host:<12} {outcome}")
            if args.action == "status" and not args.no_progress:
                show_progress(variants)


if __name__ == "__main__":
    main()
