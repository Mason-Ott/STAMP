"""Compare sampling strategies: one config per strategy, each run on its own host.

    python scripts/sampling_sweep.py configs/crc_pipeline.yaml start --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml status --n-tiles 1024
    python scripts/sampling_sweep.py configs/crc_pipeline.yaml stop --n-tiles 1024

Actions: `configs` (only write the configs and show the plan), `start`, `status`,
`stop`. Run it from the STAMP folder, with the environment active.

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
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

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
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    variants = plan_variants(base, args.strategies, args.n_tiles, args.hosts)
    out_dir = (
        args.out_dir or Path("configs/sweeps") / f"sampling_{args.n_tiles}"
    ).resolve()
    paths = write_configs(variants, out_dir)

    for variant in variants:
        path = paths[variant.strategy]
        print(f"\n== {variant.strategy} on {variant.host}: {path}")
        if args.action == "configs":
            print(f"   steps: {variant.steps}")
            continue
        command = [
            sys.executable,
            "-m",
            "stamp",
            "-c",
            str(path),
            "workers",
            args.action,
        ]
        if args.action == "start":
            command += ["--steps", variant.steps]
        if args.action == "stop" and args.force:
            command.append("--force")
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
