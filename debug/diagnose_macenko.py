"""
Macenko stain-normalization experiment.

Fits Macenko stain vectors from a reference tile (taken from whichever
dataset you consider "canonical" -- default: the good/camelyon one),
then normalizes sampled tiles from BOTH datasets to that reference.

Compares RGB statistics before vs after normalization, per dataset,
to see whether normalization actually closes the color gap we found
in diagnose_canny.py / diagnose_datasets.py.

No external stain-normalization library required -- implements the
classic Macenko et al. (2009) algorithm directly with numpy.

Usage:
    python diagnose_macenko.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DATASET_A_NAME = "camelyon_tif"
DATASET_A_DIR = Path("/csse/research/cancerimages/camelyon16/wsis")
DATASET_A_GLOB = "*.tif"

DATASET_B_NAME = "crc_svs"
DATASET_B_DIR = Path("/csse/research/cancerimages/CRC_Dataset/wsis")
DATASET_B_GLOB = "*.svs"

# Which slide/location to take the REFERENCE stain vectors from.
# Default: first camelyon slide. Change if you want a CRC reference instead.
REFERENCE_DATASET = DATASET_A_NAME
REFERENCE_SLIDE_INDEX = 0  # index into sorted glob results

N_SLIDES_PER_DATASET = 3
N_TILES_PER_SLIDE = 30

TILE_SIZE_UM = 256.0
TILE_SIZE_PX = 224

BRIGHTNESS_CUTOFF = 240
CANNY_CUTOFF = 0.02

OUTPUT_DIR = Path("/csse/research/cancerimages/diagnostic_output_macenko")

# Macenko params
OD_THRESHOLD = 0.15  # optical density threshold to discard background pixels
ANGULAR_PERCENTILE = 1  # percentile for extreme angle selection (1st/99th)


# ----------------------------------------------------------------------
# Macenko stain normalization (classic implementation)
# ----------------------------------------------------------------------


def rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to optical density."""
    img = img.astype(np.float64)
    img[img == 0] = 1.0  # avoid log(0)
    return -np.log(img / 255.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def get_stain_matrix(img: np.ndarray, od_threshold: float = OD_THRESHOLD) -> np.ndarray:
    """
    Estimate the 2 stain vectors (H&E) from an RGB tile using Macenko's method.
    Returns a 3x2 matrix (columns = stain vectors, in OD space).
    """
    od = rgb_to_od(img).reshape(-1, 3)
    # remove background / low-OD pixels
    od_thresh = od[np.all(od > od_threshold, axis=1)]
    if od_thresh.shape[0] < 10:
        # fall back to all pixels if too few remain
        od_thresh = od

    # eigenvectors of covariance
    cov = np.cov(od_thresh.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # take the two largest eigenvectors (last two columns, ascending order)
    top2 = eigvecs[:, [-1, -2]]

    # project onto plane spanned by top2, compute angles
    proj = od_thresh @ top2
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    min_angle = np.percentile(angles, ANGULAR_PERCENTILE)
    max_angle = np.percentile(angles, 100 - ANGULAR_PERCENTILE)

    v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

    # order so that Hematoxylin (more red/purple, typically larger R OD) comes first
    if v_min[0] > v_max[0]:
        stain_matrix = np.stack([v_min, v_max], axis=1)
    else:
        stain_matrix = np.stack([v_max, v_min], axis=1)

    # normalize columns to unit length
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)
    return stain_matrix  # 3x2


def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    od = rgb_to_od(img).reshape(-1, 3).T  # 3 x N
    # solve stain_matrix (3x2) @ concentrations (2xN) = od (3xN) via least squares
    concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od, rcond=None)
    return concentrations  # 2 x N


def normalize_to_reference(
    img: np.ndarray,
    ref_stain_matrix: np.ndarray,
    ref_max_conc: np.ndarray,
) -> np.ndarray:
    """Normalize `img` so its stain concentrations match the reference's max concentrations."""
    src_stain_matrix = get_stain_matrix(img)
    src_conc = get_concentrations(img, src_stain_matrix)  # 2 x N

    src_max_conc = np.percentile(src_conc, 99, axis=1, keepdims=True)
    src_max_conc[src_max_conc == 0] = 1e-6

    scaled_conc = src_conc * (ref_max_conc / src_max_conc)

    od_normalized = ref_stain_matrix @ scaled_conc  # 3 x N
    od_normalized = od_normalized.T.reshape(img.shape)
    return od_to_rgb(od_normalized)


# ----------------------------------------------------------------------
# Tile sampling (same approach as previous scripts)
# ----------------------------------------------------------------------


@dataclass
class SampledTile:
    dataset: str
    slide_name: str
    rgb: np.ndarray  # original uint8 tile


def get_mpp(slide: openslide.AbstractSlide) -> float | None:
    val = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    return float(val) if val is not None else None


def edge_score_of(tile_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    return float(np.array(edges).mean() / 255)


def sample_tiles(slide_path: Path, dataset_name: str, n_tiles: int) -> list[SampledTile]:
    slide = openslide.open_slide(str(slide_path))
    mpp = get_mpp(slide)
    if mpp is None:
        return []
    tile_size_slide_px = int(round(TILE_SIZE_UM / mpp))
    width, height = slide.dimensions
    rng = np.random.default_rng(seed=7)

    out: list[SampledTile] = []
    tried = 0
    while len(out) < n_tiles and tried < n_tiles * 30:
        tried += 1
        x = int(rng.integers(0, max(1, width - tile_size_slide_px)))
        y = int(rng.integers(0, max(1, height - tile_size_slide_px)))
        region = slide.read_region((x, y), 0, (tile_size_slide_px, tile_size_slide_px))
        region = region.resize((TILE_SIZE_PX, TILE_SIZE_PX)).convert("RGB")
        tile_np = np.array(region)

        gray_mean = float(cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY).mean())
        if BRIGHTNESS_CUTOFF is not None and gray_mean >= BRIGHTNESS_CUTOFF:
            continue
        if CANNY_CUTOFF is not None and edge_score_of(tile_np) < CANNY_CUTOFF:
            continue

        out.append(SampledTile(dataset=dataset_name, slide_name=slide_path.name, rgb=tile_np))

    print(f"  {slide_path.name}: sampled {len(out)}/{n_tiles} tiles ({tried} tries)")
    return out


def rgb_stats(tiles: list[np.ndarray]) -> dict:
    if not tiles:
        return {}
    means = np.array([t.astype(np.float64).mean(axis=(0, 1)) for t in tiles])
    stds = np.array([t.astype(np.float64).std(axis=(0, 1)) for t in tiles])
    return {
        "mean_of_means": means.mean(axis=0).tolist(),
        "std_of_means": means.std(axis=0).tolist(),
        "mean_of_stds": stds.mean(axis=0).tolist(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gather reference tile + fit reference stain matrix
    ref_dir = DATASET_A_DIR if REFERENCE_DATASET == DATASET_A_NAME else DATASET_B_DIR
    ref_glob = DATASET_A_GLOB if REFERENCE_DATASET == DATASET_A_NAME else DATASET_B_GLOB
    ref_slide_path = sorted(ref_dir.glob(ref_glob))[REFERENCE_SLIDE_INDEX]
    print(f"Using reference slide: {ref_slide_path.name} ({REFERENCE_DATASET})")

    ref_tiles = sample_tiles(ref_slide_path, REFERENCE_DATASET, 5)
    if not ref_tiles:
        raise RuntimeError("Could not sample a reference tile -- check paths/MPP.")
    # use the most texture-rich reference tile (best stain signal)
    ref_tile = max(ref_tiles, key=lambda t: edge_score_of(t.rgb)).rgb

    ref_stain_matrix = get_stain_matrix(ref_tile)
    ref_conc = get_concentrations(ref_tile, ref_stain_matrix)
    ref_max_conc = np.percentile(ref_conc, 99, axis=1, keepdims=True)

    Image.fromarray(ref_tile).save(OUTPUT_DIR / "reference_tile.png")
    print(f"Reference stain matrix:\n{ref_stain_matrix}")
    print(f"Reference max concentrations:\n{ref_max_conc.ravel()}")

    # 2. Sample tiles from both datasets
    all_tiles: list[SampledTile] = []
    for dataset_name, dataset_dir, glob_pat in [
        (DATASET_A_NAME, DATASET_A_DIR, DATASET_A_GLOB),
        (DATASET_B_NAME, DATASET_B_DIR, DATASET_B_GLOB),
    ]:
        print(f"\n=== sampling {dataset_name} ===")
        for slide_path in sorted(dataset_dir.glob(glob_pat))[:N_SLIDES_PER_DATASET]:
            all_tiles.extend(sample_tiles(slide_path, dataset_name, N_TILES_PER_SLIDE))

    # 3. Normalize each tile to the reference, track successes/failures
    normalized_by_dataset: dict[str, list[np.ndarray]] = {DATASET_A_NAME: [], DATASET_B_NAME: []}
    original_by_dataset: dict[str, list[np.ndarray]] = {DATASET_A_NAME: [], DATASET_B_NAME: []}
    n_failed = {DATASET_A_NAME: 0, DATASET_B_NAME: 0}

    saved_examples = {DATASET_A_NAME: 0, DATASET_B_NAME: 0}

    for tile in all_tiles:
        original_by_dataset[tile.dataset].append(tile.rgb)
        try:
            norm = normalize_to_reference(tile.rgb, ref_stain_matrix, ref_max_conc)
            normalized_by_dataset[tile.dataset].append(norm)

            if saved_examples[tile.dataset] < 3:
                side_by_side = np.concatenate([tile.rgb, norm], axis=1)
                Image.fromarray(side_by_side).save(
                    OUTPUT_DIR / f"{tile.dataset}_before_after_{saved_examples[tile.dataset]}.png"
                )
                saved_examples[tile.dataset] += 1
        except Exception as e:
            n_failed[tile.dataset] += 1

    # 4. Compare stats before/after per dataset
    print("\n=== RGB STATS: BEFORE vs AFTER NORMALIZATION ===")
    results = {}
    for dataset_name in [DATASET_A_NAME, DATASET_B_NAME]:
        before = rgb_stats(original_by_dataset[dataset_name])
        after = rgb_stats(normalized_by_dataset[dataset_name])
        results[dataset_name] = {
            "n_tiles": len(original_by_dataset[dataset_name]),
            "n_failed_normalization": n_failed[dataset_name],
            "before": before,
            "after": after,
        }
        print(f"\n{dataset_name}:")
        print(f"  before mean RGB: {before.get('mean_of_means')}")
        print(f"  after  mean RGB: {after.get('mean_of_means')}")
        print(f"  before std-of-means (inter-tile variability): {before.get('std_of_means')}")
        print(f"  after  std-of-means (inter-tile variability): {after.get('std_of_means')}")

    # Cross-dataset gap before vs after
    if results[DATASET_A_NAME]["before"] and results[DATASET_B_NAME]["before"]:
        before_gap = np.array(results[DATASET_A_NAME]["before"]["mean_of_means"]) - np.array(
            results[DATASET_B_NAME]["before"]["mean_of_means"]
        )
        after_gap = np.array(results[DATASET_A_NAME]["after"]["mean_of_means"]) - np.array(
            results[DATASET_B_NAME]["after"]["mean_of_means"]
        )
        print(f"\nCross-dataset RGB-mean gap BEFORE normalization: {before_gap.tolist()}")
        print(f"Cross-dataset RGB-mean gap AFTER  normalization: {after_gap.tolist()}")
        print(f"Gap magnitude before: {np.linalg.norm(before_gap):.2f}")
        print(f"Gap magnitude after:  {np.linalg.norm(after_gap):.2f}")

    with open(OUTPUT_DIR / "macenko_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBefore/after example images + results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()