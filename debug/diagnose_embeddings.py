"""
CONCH embedding separability diagnostic.

For labeled slides in both datasets, extracts real CONCH embeddings from
tissue tiles, then compares:

  1. Within-slide tile similarity (baseline noise level)
  2. Within-class, cross-slide similarity (same label, different slide)
  3. Between-class similarity (different label)

If (2) is not meaningfully higher than (3), CONCH embeddings are not
separating the classes for that dataset -- a much more direct explanation
of a low downstream accuracy than pixel/stain statistics.

Also runs a simple leave-one-slide-out nearest-centroid probe as a sanity
check on separability, per dataset.

Usage:
    python diagnose_embeddings.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from PIL import Image

OUTPUT_DIR = Path("/csse/research/cancerimages/diagnostic_output_embeddings")

# ----------------------------------------------------------------------
# CONFIG -- labeled slides
# ----------------------------------------------------------------------

CAMELYON_DIR = Path("/csse/research/cancerimages/camelyon16/wsis")
CRC_DIR = Path("/csse/research/cancerimages/CRC_Dataset/wsis")

# dataset -> list of (slide_stem, label)  label: 1 = positive, 0 = negative
CAMELYON_SLIDES = [
    ("normal_002", 0),
    ("normal_011", 0),
    ("normal_016", 0),
    ("tumor_089", 1),
    ("tumor_056", 1),
    ("tumor_016", 1),
]
CAMELYON_EXT = ".tif"

# CRC: pulled from dataset.xlsx (StudyID/SlideID -> isLNM). Fill in from your sheet;
# these six mirror the rows you showed (3 positive / 3 negative).
CRC_SLIDES = [
    ("LNM_A0002", 1),
    ("LNM_A0003", 1),
    ("LNM_A0004", 1),
    ("LNM_A0008", 0),
    ("LNM_A0009", 0),
    ("LNM_A0010", 0),
]
CRC_EXT = ".svs"

N_TILES_PER_SLIDE = 20
TILE_SIZE_UM = 256.0
TILE_SIZE_PX = 224
BRIGHTNESS_CUTOFF = 240
CANNY_CUTOFF = 0.02

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Tiling
# ----------------------------------------------------------------------


def get_mpp(slide: openslide.AbstractSlide) -> float | None:
    val = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    return float(val) if val is not None else None


def edge_score_of(tile_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    return float(np.array(edges).mean() / 255)


def sample_tiles(slide_path: Path, n_tiles: int) -> list[Image.Image]:
    slide = openslide.open_slide(str(slide_path))
    mpp = get_mpp(slide)
    if mpp is None:
        print(f"  WARNING: no MPP for {slide_path.name}")
        return []
    tile_size_slide_px = int(round(TILE_SIZE_UM / mpp))
    width, height = slide.dimensions
    rng = np.random.default_rng(seed=0)

    tiles: list[Image.Image] = []
    tried = 0
    while len(tiles) < n_tiles and tried < n_tiles * 40:
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
        tiles.append(region)

    print(f"  {slide_path.name}: {len(tiles)}/{n_tiles} tiles ({tried} tries)")
    return tiles


# ----------------------------------------------------------------------
# CONCH embeddings
# ----------------------------------------------------------------------


def load_conch():
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
    )
    model = model.to(DEVICE).eval()
    return model, preprocess


@torch.no_grad()
def embed_tiles(model, preprocess, tiles: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    embeddings = []
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i : i + batch_size]
        tensor_batch = torch.stack([preprocess(t) for t in batch]).to(DEVICE)
        emb = model.encode_image(tensor_batch, proj_contrast=False, normalize=False)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 512))


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


@dataclass
class TileEmbedding:
    dataset: str
    slide: str
    label: int
    vec: np.ndarray


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def analyze_dataset(records: list[TileEmbedding], dataset_name: str) -> dict:
    slides = sorted(set(r.slide for r in records))
    by_slide = {s: [r for r in records if r.slide == s] for s in slides}

    # 1. within-slide similarity
    within_slide_sims = []
    for s in slides:
        vecs = np.array([r.vec for r in by_slide[s]])
        if len(vecs) < 2:
            continue
        sim = cosine_sim_matrix(vecs, vecs)
        iu = np.triu_indices_from(sim, k=1)
        within_slide_sims.extend(sim[iu].tolist())

    # 2. within-class, cross-slide + 3. between-class
    within_class_cross_slide = []
    between_class = []
    for s1 in slides:
        for s2 in slides:
            if s1 >= s2:
                continue  # each pair once
            label1 = by_slide[s1][0].label
            label2 = by_slide[s2][0].label
            vecs1 = np.array([r.vec for r in by_slide[s1]])
            vecs2 = np.array([r.vec for r in by_slide[s2]])
            sim = cosine_sim_matrix(vecs1, vecs2).ravel().tolist()
            if label1 == label2:
                within_class_cross_slide.extend(sim)
            else:
                between_class.extend(sim)

    # Leave-one-slide-out nearest-centroid probe
    correct = 0
    total = 0
    for held_out in slides:
        held_out_label = by_slide[held_out][0].label
        held_out_vecs = np.array([r.vec for r in by_slide[held_out]])
        held_out_mean = held_out_vecs.mean(axis=0)

        train_slides = [s for s in slides if s != held_out]
        centroids = {0: [], 1: []}
        for s in train_slides:
            vecs = np.array([r.vec for r in by_slide[s]])
            centroids[by_slide[s][0].label].append(vecs.mean(axis=0))
        if not centroids[0] or not centroids[1]:
            continue
        c0 = np.mean(centroids[0], axis=0)
        c1 = np.mean(centroids[1], axis=0)

        d0 = np.linalg.norm(held_out_mean - c0)
        d1 = np.linalg.norm(held_out_mean - c1)
        pred = 0 if d0 < d1 else 1
        correct += int(pred == held_out_label)
        total += 1

    result = {
        "n_slides": len(slides),
        "n_tiles": len(records),
        "within_slide_sim_mean": float(np.mean(within_slide_sims)) if within_slide_sims else None,
        "within_class_cross_slide_sim_mean": (
            float(np.mean(within_class_cross_slide)) if within_class_cross_slide else None
        ),
        "between_class_sim_mean": float(np.mean(between_class)) if between_class else None,
        "separation_margin": (
            float(np.mean(within_class_cross_slide) - np.mean(between_class))
            if within_class_cross_slide and between_class
            else None
        ),
        "loso_centroid_accuracy": correct / total if total else None,
        "loso_n_slides_evaluated": total,
    }
    print(f"\n=== {dataset_name} ===")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading CONCH on {DEVICE}...")
    model, preprocess = load_conch()

    all_embeddings: list[TileEmbedding] = []

    for dataset_name, base_dir, ext, slide_list in [
        ("camelyon_tif", CAMELYON_DIR, CAMELYON_EXT, CAMELYON_SLIDES),
        ("crc_svs", CRC_DIR, CRC_EXT, CRC_SLIDES),
    ]:
        print(f"\n=== extracting embeddings: {dataset_name} ===")
        for stem, label in slide_list:
            slide_path = base_dir / f"{stem}{ext}"
            if not slide_path.exists():
                print(f"  WARNING: {slide_path} not found, skipping")
                continue
            tiles = sample_tiles(slide_path, N_TILES_PER_SLIDE)
            if not tiles:
                continue
            embs = embed_tiles(model, preprocess, tiles)
            for vec in embs:
                all_embeddings.append(
                    TileEmbedding(dataset=dataset_name, slide=stem, label=label, vec=vec)
                )

    results = {}
    for dataset_name in ["camelyon_tif", "crc_svs"]:
        recs = [r for r in all_embeddings if r.dataset == dataset_name]
        if not recs:
            print(f"No embeddings for {dataset_name}, skipping analysis")
            continue
        results[dataset_name] = analyze_dataset(recs, dataset_name)

    with open(OUTPUT_DIR / "embedding_separability.json", "w") as f:
        json.dump(results, f, indent=2)

    # save raw embeddings too, in case further analysis is wanted
    np.savez(
        OUTPUT_DIR / "raw_embeddings.npz",
        vecs=np.array([r.vec for r in all_embeddings]),
        datasets=np.array([r.dataset for r in all_embeddings]),
        slides=np.array([r.slide for r in all_embeddings]),
        labels=np.array([r.label for r in all_embeddings]),
    )

    print(f"\nResults + raw embeddings saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()