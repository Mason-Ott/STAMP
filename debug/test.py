"""
Diagnose the background-rejection (canny + brightness) cutoffs.

Unlike diagnose_datasets.py, this script does NOT discard rejected tiles.
It samples a dense/random set of candidate tile locations per slide,
computes brightness + edge_score for EVERY candidate (accepted or not),
and:

  1. Reports the full distribution of edge_score and brightness per dataset
     (so you can see if the two datasets need different cutoffs).
  2. Sweeps a range of canny cutoffs and reports acceptance rate + resulting
     RGB stats of the accepted tiles, for both datasets side by side.
  3. Saves histograms (as PNG) of edge_score and brightness per dataset.

Usage:
    python diagnose_canny.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openslide

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DATASET_A_NAME = "camelyon_tif"
DATASET_A_DIR = Path("/csse/research/cancerimages/camelyon16/wsis")
DATASET_A_GLOB = "*.tif"

DATASET_B_NAME = "crc_svs"
DATASET_B_DIR = Path("/csse/research/cancerimages/CRC_Dataset/wsis")
DATASET_B_GLOB = "*.svs"

N_SLIDES_PER_DATASET = 3
N_CANDIDATES_PER_SLIDE = 300  # unfiltered candidate tiles per slide

TILE_SIZE_UM = 256.0
TILE_SIZE_PX = 224

CANNY_CUTOFF_SWEEP = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
BRIGHTNESS_CUTOFF = 240  # applied before canny, matches STAMP default; None to disable

OUTPUT_DIR = Path("/csse/research/cancerimages/diagnostic_output_canny")


# ----------------------------------------------------------------------


@dataclass
class Candidate:
    dataset: str
    slide_name: str
    brightness_mean: float
    edge_score: float
    rgb_mean: list[float]


def get_mpp(slide: openslide.AbstractSlide) -> float | None:
    val = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    return float(val) if val is not None else None


def edge_score_of(tile_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    return float(np.array(edges).mean() / 255)


def sample_candidates(
    slide_path: Path, dataset_name: str, n_candidates: int
) -> list[Candidate]:
    slide = openslide.open_slide(str(slide_path))
    mpp = get_mpp(slide)
    if mpp is None:
        print(f"  WARNING: no MPP for {slide_path.name}, skipping")
        return []

    tile_size_slide_px = int(round(TILE_SIZE_UM / mpp))
    width, height = slide.dimensions
    rng = np.random.default_rng(seed=123)

    candidates: list[Candidate] = []
    for _ in range(n_candidates):
        x = int(rng.integers(0, max(1, width - tile_size_slide_px)))
        y = int(rng.integers(0, max(1, height - tile_size_slide_px)))

        region = slide.read_region((x, y), 0, (tile_size_slide_px, tile_size_slide_px))
        region = region.resize((TILE_SIZE_PX, TILE_SIZE_PX)).convert("RGB")
        tile_np = np.array(region)

        gray_mean = float(cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY).mean())
        escore = edge_score_of(tile_np)
        rgb_mean = tile_np.astype(np.float64).mean(axis=(0, 1)).tolist()

        candidates.append(
            Candidate(
                dataset=dataset_name,
                slide_name=slide_path.name,
                brightness_mean=gray_mean,
                edge_score=escore,
                rgb_mean=rgb_mean,
            )
        )

    print(f"  {slide_path.name}: mpp={mpp:.4f}, sampled {len(candidates)} raw candidates")
    return candidates


def sweep_cutoffs(candidates: list[Candidate], dataset_name: str) -> list[dict]:
    results = []
    for cutoff in CANNY_CUTOFF_SWEEP:
        kept = [
            c
            for c in candidates
            if (BRIGHTNESS_CUTOFF is None or c.brightness_mean < BRIGHTNESS_CUTOFF)
            and c.edge_score >= cutoff
        ]
        frac_kept = len(kept) / len(candidates) if candidates else 0.0
        rgb_mean = (
            np.array([k.rgb_mean for k in kept]).mean(axis=0).tolist() if kept else None
        )
        results.append(
            {
                "dataset": dataset_name,
                "canny_cutoff": cutoff,
                "n_candidates": len(candidates),
                "n_kept": len(kept),
                "frac_kept": frac_kept,
                "rgb_mean_of_kept": rgb_mean,
            }
        )
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_candidates: list[Candidate] = []

    for dataset_name, dataset_dir, glob_pat in [
        (DATASET_A_NAME, DATASET_A_DIR, DATASET_A_GLOB),
        (DATASET_B_NAME, DATASET_B_DIR, DATASET_B_GLOB),
    ]:
        print(f"\n=== {dataset_name} ===")
        slide_paths = sorted(dataset_dir.glob(glob_pat))[:N_SLIDES_PER_DATASET]
        for slide_path in slide_paths:
            all_candidates.extend(
                sample_candidates(slide_path, dataset_name, N_CANDIDATES_PER_SLIDE)
            )

    # Raw distribution stats (unfiltered) per dataset
    print("\n=== RAW (UNFILTERED) DISTRIBUTIONS ===")
    dist_summary = {}
    for dataset_name in [DATASET_A_NAME, DATASET_B_NAME]:
        cands = [c for c in all_candidates if c.dataset == dataset_name]
        brightness = np.array([c.brightness_mean for c in cands])
        edges = np.array([c.edge_score for c in cands])
        dist_summary[dataset_name] = {
            "n": len(cands),
            "brightness_percentiles": {
                str(p): float(np.percentile(brightness, p))
                for p in [1, 5, 25, 50, 75, 95, 99]
            },
            "edge_score_percentiles": {
                str(p): float(np.percentile(edges, p))
                for p in [1, 5, 25, 50, 75, 95, 99]
            },
            "frac_above_brightness_cutoff": (
                float((brightness >= BRIGHTNESS_CUTOFF).mean())
                if BRIGHTNESS_CUTOFF
                else None
            ),
        }
    print(json.dumps(dist_summary, indent=2))

    # Cutoff sweep
    print("\n=== CANNY CUTOFF SWEEP ===")
    sweep_results = []
    for dataset_name in [DATASET_A_NAME, DATASET_B_NAME]:
        cands = [c for c in all_candidates if c.dataset == dataset_name]
        sweep_results.extend(sweep_cutoffs(cands, dataset_name))
    for r in sweep_results:
        print(
            f"{r['dataset']:15s} cutoff={r['canny_cutoff']:.2f}  "
            f"kept={r['n_kept']:4d}/{r['n_candidates']:4d} "
            f"({r['frac_kept']*100:5.1f}%)  "
            f"rgb_mean={r['rgb_mean_of_kept']}"
        )

    with open(OUTPUT_DIR / "distribution_summary.json", "w") as f:
        json.dump(dist_summary, f, indent=2)
    with open(OUTPUT_DIR / "cutoff_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    # Histograms
    if HAVE_MPL:
        for metric_name, getter in [
            ("brightness", lambda c: c.brightness_mean),
            ("edge_score", lambda c: c.edge_score),
        ]:
            plt.figure(figsize=(7, 4))
            for dataset_name in [DATASET_A_NAME, DATASET_B_NAME]:
                vals = [getter(c) for c in all_candidates if c.dataset == dataset_name]
                plt.hist(vals, bins=40, alpha=0.5, label=dataset_name, density=True)
            plt.axvline(
                BRIGHTNESS_CUTOFF if metric_name == "brightness" else CANNY_CUTOFF_SWEEP[2],
                color="black",
                linestyle="--",
                linewidth=1,
                label="default cutoff",
            )
            plt.title(f"{metric_name} distribution (unfiltered candidates)")
            plt.xlabel(metric_name)
            plt.ylabel("density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{metric_name}_histogram.png", dpi=120)
            plt.close()
        print(f"\nHistograms saved to {OUTPUT_DIR}")
    else:
        print("\nmatplotlib not available -- skipped histogram plots")

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()