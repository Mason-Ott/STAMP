"""Write one config per sampling strategy from a base config, for comparing which
tiles each strategy selects (no feature extraction or encoding, except where a
strategy needs features).

    python scripts/make_sampling_configs.py base.yaml configs_out/ --n-tiles 1024

Prints the `stamp preprocess` command for each config. All configs share the base
config's output_dir, so the tiling and combining stages are done once between them.
"""

import argparse
from pathlib import Path

import yaml

# strategy -> steps to run for it
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_config", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n-tiles", type=int, default=1024)
    parser.add_argument("--strategies", nargs="+", default=list(STEPS), choices=STEPS)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for strategy in args.strategies:
        config = yaml.safe_load(yaml.safe_dump(base))  # deep copy
        config["preprocessing"]["save_visualisations"] = True
        pipeline = config["pipeline"]
        for key in ("encoder", "workers", "steps", "slide_feat_dir"):
            pipeline.pop(key, None)
        pipeline["order"] = (
            "extract_then_sample"
            if strategy == "conch_kmeans"
            else "sample_then_extract"
        )
        pipeline["sampling"] = {
            **pipeline.get("sampling", {}),
            "strategy": strategy,
            "n_tiles": args.n_tiles,
        }
        path = args.out_dir / f"sampling_{strategy}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"stamp -c {path} preprocess --steps {STEPS[strategy]}")


if __name__ == "__main__":
    main()
