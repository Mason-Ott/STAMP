"""Compare sampling strategies: one config per strategy, each run on its own host.

    python scripts/sampling_sweep.py configs/crc_pipeline.yaml start --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml status --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml stop --n-tiles 1024

By default the sweep only samples (positions, thumbnails). `--steps
tile,combine,sample,extract` also extracts the features of the sampled tiles
(give the same `--steps` to `status`).

Run it from the STAMP folder, with the environment active.

Actions
  start      give every strategy a host and start its worker there.
  status     worker state and progress of every strategy.
  stop       stop the workers (--force kills them).
  configs    only write the configs and show the plan (no ssh).

`pipeline.workers.hosts` is a **pool**. `start` goes through the strategies in
order and, for each, tries the pool's hosts in order (skipping hosts already
given a strategy) until one starts it: it just runs `ssh host ...`. A host that
accepts your key needs nothing. A host that asks for a password asks you on the
terminal: type it, or press Enter to skip that host. If a host fails for any
reason (no such host, wrong password, unreachable) the next one is tried. Put more
hosts in the pool than strategies to have spares: once every strategy has a host,
the spare hosts join the most expensive runs as helpers (extracting features,
reading tile images), taking turns. They share a run's work through lock files, so
nothing is done twice (`--no-helpers` leaves them idle). If the pool runs out, the
strategies still without a host are listed with the command to start them.

The connection stays open for a few hours, so `status` and `stop` do not ask for
the password again. Each strategy gets its own config: a copy of the base config
with the strategy and `n_tiles` set, no encoder, visualisations on, and
`workers.hosts` set to its host. They are written to `--out-dir` (default
`configs/sweeps/<experiment>`, which must be on storage every host sees) and
started with `stamp workers start`, so logs and progress work as for any run
(`<output_dir>/logs/<host>.log`). `status` and `stop` use the hosts recorded there.

The sweep is one experiment (`pipeline.experiment` in the base config) with one run
per strategy: `<output_dir>/<experiment>/<strategy>/` holds that run's config.yaml,
README.md, sampled tiles, thumbnails and logs. Tiling and combining are done once
for all runs, in `<output_dir>/<experiment>/shared/` (the hosts split the slides).
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


COST_ORDER = [
    "conch_kmeans",  # extracts features of every tile on a GPU
    "texture",
    "yottixel",
    "entropy",
    "tissue",  # these four read every tile image
    "spatial",
    "random",
    "first_n",  # positions only
]
"""Strategies, most expensive first: spare hosts help in this order."""


@dataclass(frozen=True)
class Variant:
    strategy: str
    host: str
    steps: str
    config: dict
    helpers: tuple[str, ...] = ()
    """Further hosts working on the same run (see `add_helpers`)."""

    @property
    def hosts(self) -> list[str]:
        return [self.host, *self.helpers] if self.host else []


# --- planning (no ssh) ---------------------------------------------------------


def plan_variants(
    base: dict,
    strategies: list[str],
    n_tiles: int,
    hosts: list[str] | None = None,
    steps: str | None = None,
) -> list[Variant]:
    """One variant per strategy, assigned to `hosts` (default: the config's pool)
    in turn. `steps` (comma separated) replaces the default `STEPS` of every
    strategy."""
    workers = base.get("pipeline", {}).get("workers")
    if not workers or not (hosts or workers.get("hosts")):
        raise ValueError("the config needs pipeline.workers.hosts (or pass --hosts)")
    if not base["pipeline"].get("experiment"):
        raise ValueError("the config needs pipeline.experiment (the sweep's name)")
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
        pipeline["run"] = strategy
        # workers waiting for each other's items check often: the items are short
        pipeline["coordination"] = {
            "poll_interval_s": 5.0,
            **pipeline.get("coordination", {}),
        }
        pipeline["sampling"] = {
            **pipeline.get("sampling", {}),
            "strategy": strategy,
            "n_tiles": n_tiles,
        }
        variants.append(Variant(strategy, "", steps or STEPS[strategy], config))
    return [with_host(v, hosts[i % len(hosts)]) for i, v in enumerate(variants)]


def with_host(variant: Variant, host: str) -> Variant:
    return _with_hosts(variant, host, ())


def with_helpers(variant: Variant, helpers: list[str]) -> Variant:
    return _with_hosts(variant, variant.host, tuple(helpers))


def _with_hosts(variant: Variant, host: str, helpers: tuple[str, ...]) -> Variant:
    config = yaml.safe_load(yaml.safe_dump(variant.config))
    workers = config["pipeline"]["workers"]
    config["pipeline"]["workers"] = {**workers, "hosts": [host, *helpers]}
    return dataclasses.replace(variant, host=host, helpers=helpers, config=config)


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


def load_variants(
    out_dir: Path, strategies: list[str], steps: str | None = None
) -> list[Variant]:
    """The variants as written by `start`/`configs` (with the hosts they got).
    Pass the `steps` given to `start` so `status` counts them."""
    variants = []
    for strategy in strategies:
        path = config_path(out_dir, strategy)
        if not path.exists():
            raise SystemExit(f"{path} does not exist: run `start` first")
        config = yaml.safe_load(path.read_text())
        host, *helpers = config["pipeline"]["workers"]["hosts"]
        variants.append(
            Variant(strategy, host, steps or STEPS[strategy], config, tuple(helpers))
        )
    return variants


def allocate(
    variants: list[Variant],
    pool: list[str],
    starter: Callable[[Variant], str],
    report: Callable[[Variant, str], None] = lambda variant, outcome: None,
) -> list[tuple[Variant, str]]:
    """Give each variant, in order, the first unused host of the pool that starts
    it (`starter` returns an outcome; one starting with "failed" moves on to the
    next host). A host is never given to two variants. Returns each variant, with
    the host it ended up on, and its last outcome; a variant left without a host
    has host "" and a failed outcome. `report` is called for every failed try."""
    free = list(pool)
    results = []
    for variant in variants:
        placed, outcome = with_host(variant, ""), "failed: no host left in the pool"
        while free:
            placed = with_host(variant, free.pop(0))
            outcome = starter(placed)
            if not is_failure(outcome):
                break
            report(placed, outcome)
        results.append((placed, outcome))
    return results


def add_helpers(
    placed: list[Variant],
    spares: list[str],
    starter: Callable[[Variant, str], str],
    report: Callable[[str, str], None] = lambda host, outcome: None,
    limit: int | None = None,
) -> dict[str, list[str]]:
    """Let the spare hosts help the runs that have a host, most expensive run
    first, taking turns. A helper runs the same config next to the run's own
    worker: they share the work through lock files, so each item (patient, slide)
    is done once, by whoever gets there first. A host that fails is skipped. At
    most `limit` helpers are started (None: as many as there are spare hosts).
    Returns strategy -> helper hosts."""
    ranked = sorted(placed, key=lambda v: COST_ORDER.index(v.strategy))
    helpers: dict[str, list[str]] = {v.strategy: [] for v in ranked}
    for host in spares:
        n_helpers = sum(map(len, helpers.values()))
        if not ranked or (limit is not None and n_helpers >= limit):
            break
        target = ranked[n_helpers % len(ranked)]
        outcome = starter(target, host)
        if is_failure(outcome):
            report(host, outcome)
        else:
            helpers[target.strategy].append(host)
    return helpers


def is_failure(outcome: str) -> bool:
    return outcome.startswith("failed")


# --- ssh -----------------------------------------------------------------------


def remote_run(variant: Variant, config_file: Path) -> RemoteRun:
    workers = variant.config["pipeline"]["workers"]
    return RemoteRun(
        config_file=config_file,
        # the run's own folder: its logs and the worker's session name come from it
        output_dir=(
            Path(variant.config["preprocessing"]["output_dir"])
            / variant.config["pipeline"]["experiment"]
            / variant.config["pipeline"]["run"]
        ).resolve(),
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


def parallel[T, R](function: Callable[[T], R], items: list[T]) -> list[R]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(len(items), 32)) as pool:
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


def parse_steps(text: str) -> str:
    """Check a `--steps` value: comma separated pipeline step names."""
    from stamp.preprocessing.pipeline.config import StepName

    names = [name.strip() for name in text.split(",")]
    unknown = [name for name in names if name not in list(StepName)]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown step(s) {', '.join(unknown)}; choose from {', '.join(StepName)}"
        )
    return ",".join(names)


def print_plan(variants: list[Variant], out_dir: Path) -> None:
    print(f"configs in {out_dir}")
    for variant in variants:
        print(f"  {variant.strategy:<13} {variant.host:<12} steps: {variant.steps}")


def host_report(pool: list[str], problems: dict[str, str | None]) -> list[str]:
    """Lines saying which pool hosts work without a password and why the others
    don't, with what to do about each group."""
    groups: dict[str, list[str]] = {
        "works": [],
        "no such host": [],
        "host key changed": [],
        "needs a password": [],
        "other": [],
    }
    for host in pool:
        why = problems[host]
        if why is None:
            groups["works"].append(host)
        elif "resolve" in why.lower():
            groups["no such host"].append(host)
        elif "host key verification failed" in why.lower():
            groups["host key changed"].append(host)
        elif "permission denied" in why.lower():
            groups["needs a password"].append(host)
        else:
            groups["other"].append(host)
    advice = {
        "no such host": "remove from the pool",
        "host key changed": "the machine may have been reinstalled: once you are sure, "
        "`ssh-keygen -R <host>`",
        "needs a password": "start with --no-prompt skips them; to use them, once "
        "per host: ssh-copy-id -i ~/.ssh/id_ed25519.pub <host>",
        "other": "unreachable or slow",
    }
    lines = [f"{len(pool)} hosts in the pool:"]
    for name, hosts in groups.items():
        if not hosts:
            continue
        lines.append(f"  {name}: {len(hosts)}")
        if name in advice:
            lines.append(f"      ({advice[name]})")
        lines.append("      " + " ".join(hosts))
    return lines


def explain_problems(problems: dict[str, str | None]) -> None:
    broken = {host: why for host, why in problems.items() if why}
    if not broken:
        return
    print("\nhosts not reachable without a password prompt:")
    for host, why in broken.items():
        print(f"  {host}: {why}")
    print("  (hosts started through `start` stay connected for a few hours)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base_config", type=Path)
    parser.add_argument(
        "action",
        choices=["hosts", "configs", "start", "status", "stop"],
        default="configs",
    )
    parser.add_argument("--n-tiles", type=int, default=1024)
    parser.add_argument("--strategies", nargs="+", default=list(STEPS), choices=STEPS)
    parser.add_argument("--hosts", nargs="+", help="the pool (default: from config)")
    parser.add_argument(
        "--steps",
        type=parse_steps,
        help="steps to run for every strategy, e.g. tile,combine,sample,extract "
        "(default: the sampling steps of each strategy, see STEPS). Give the same "
        "value to status",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="stop: kill immediately")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="start: use only hosts that accept a passwordless login (checked in "
        "parallel first) and never ask for a password",
    )
    parser.add_argument(
        "--max-hosts",
        type=int,
        help="start: use at most this many hosts in total (runs plus helpers)",
    )
    parser.add_argument(
        "--no-helpers",
        action="store_true",
        help="start: leave spare pool hosts idle instead of helping the heaviest runs",
    )
    parser.add_argument("--no-progress", action="store_true", help="skip progress")
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    pool = args.hosts or base.get("pipeline", {}).get("workers", {}).get("hosts", [])
    if not pool:
        raise SystemExit("the config needs pipeline.workers.hosts (or pass --hosts)")
    out_dir = (
        args.out_dir
        or Path("configs/sweeps") / base.get("pipeline", {}).get("experiment", "sweep")
    ).resolve()

    match args.action:
        case "hosts":
            print("\n".join(host_report(pool, check_hosts(pool))))

        case "configs":
            variants = plan_variants(
                base, args.strategies, args.n_tiles, pool, args.steps
            )
            write_configs(variants, out_dir)
            print_plan(variants, out_dir)

        case "start":
            if args.no_prompt:
                problems = check_hosts(pool)
                skipped = [h for h in pool if problems[h]]
                pool = [h for h in pool if not problems[h]]
                print(
                    f"{len(pool)} of {len(skipped) + len(pool)} hosts accept a "
                    "passwordless login; the rest are skipped "
                    "(`hosts` action lists them)"
                )
                if not pool:
                    raise SystemExit("no host accepts a passwordless login")
            variants = plan_variants(
                base, args.strategies, args.n_tiles, pool, args.steps
            )
            print(f"{len(variants)} strategies, {len(pool)} hosts in the pool")
            paths: dict[str, Path] = {}

            def starter(variant: Variant) -> str:
                path = paths[variant.strategy] = write_configs([variant], out_dir)[
                    variant.strategy
                ]
                print(f"{variant.strategy}: trying {variant.host}")
                return start(
                    remote_run(variant, path),
                    [variant.host],
                    ["--steps", variant.steps],
                    interactive=not args.no_prompt,
                )[variant.host]

            def report(variant: Variant, outcome: str) -> None:
                print(f"  {variant.host} failed ({outcome}); trying the next host")

            results = allocate(variants, pool, starter, report)
            print()
            for variant, outcome in results:
                print(f"{variant.strategy:<13} {variant.host or '-':<12} {outcome}")

            left = [v.strategy for v, outcome in results if is_failure(outcome)]
            started = [v for v, outcome in results if not is_failure(outcome)]

            tried = {v.host for v, _ in results}
            spares = [] if args.no_helpers else [h for h in pool if h not in tried]
            if spares and started and not left:
                print(f"\n{len(spares)} spare hosts will help the heaviest runs")

                def helper_starter(variant: Variant, host: str) -> str:
                    print(f"  {host} -> {variant.strategy}")
                    return start(
                        remote_run(variant, paths[variant.strategy]),
                        [host],
                        ["--steps", variant.steps],
                        interactive=not args.no_prompt,
                    )[host]

                helpers = add_helpers(
                    started,
                    spares,
                    helper_starter,
                    lambda host, outcome: print(
                        f"  {host} failed ({outcome}); skipped"
                    ),
                    limit=(
                        None
                        if args.max_hosts is None
                        else max(0, args.max_hosts - len(started))
                    ),
                )
                for variant in started:
                    extra = helpers[variant.strategy]
                    if extra:
                        paths[variant.strategy] = write_configs(
                            [with_helpers(variant, extra)], out_dir
                        )[variant.strategy]
                        print(f"{variant.strategy:<13} + {', '.join(extra)}")
            if started:
                first = remote_run(started[0], paths[started[0].strategy])
                print(f"\nlogs: {first.log_file('<host>')}")
            if left:
                print(
                    f"\nno host left for: {', '.join(left)}.\nAdd hosts to "
                    "pipeline.workers.hosts, then start the rest on hosts not used "
                    "above:\n  python scripts/sampling_sweep.py "
                    f"{args.base_config} start --n-tiles {args.n_tiles} "
                    f"--strategies {' '.join(left)} --hosts <new hosts>"
                )
                raise SystemExit(1)

        case "status" | "stop":
            variants = load_variants(out_dir, args.strategies, args.steps)
            pairs = [(v, host) for v in variants for host in v.hosts]
            problems = check_hosts(sorted({host for _, host in pairs}))
            explain_problems(problems)
            reachable = [(v, host) for v, host in pairs if not problems[host]]
            runs = {
                v.strategy: remote_run(v, config_path(out_dir, v.strategy))
                for v in variants
            }
            if args.action == "stop":
                outcomes = parallel(
                    lambda pair: stop(
                        runs[pair[0].strategy], [pair[1]], force=args.force
                    )[pair[1]],
                    reachable,
                )
            else:
                outcomes = parallel(
                    lambda pair: (
                        "worker "
                        + host_states(runs[pair[0].strategy], [pair[1]])[pair[1]].value
                    ),
                    reachable,
                )
            result_of = {
                (variant.strategy, host): outcome
                for (variant, host), outcome in zip(reachable, outcomes, strict=True)
            }
            print()
            for variant, host in pairs:
                outcome = result_of.get((variant.strategy, host), "unreachable")
                role = "" if host == variant.host else " (helper)"
                print(f"{variant.strategy:<13} {host:<12} {outcome}{role}")
            if args.action == "status" and not args.no_progress:
                show_progress(variants)


if __name__ == "__main__":
    main()
