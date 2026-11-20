"""Step `sample`: patient bag -> patient bag of at most n_tiles tiles.

Works on bags with or without features, so it serves both orders. The tile
budget is split evenly across the patient's slides (see `allocate`); a sampling
strategy then picks that many tiles from each slide.

To add a strategy: write a function with the `Sampler` signature, add it to
SAMPLERS, and add its name to `SamplingStrategy` in pipeline/config.py.
"""

import dataclasses
import logging
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from stamp.preprocessing.data.bags import Bag, SamplingSummary, read_bag, write_bag
from stamp.preprocessing.data.clinical import Patient
from stamp.preprocessing.pipeline.config import SamplingConfig, SamplingStrategy
from stamp.preprocessing.slides.overview import save_patient_overview

_logger = logging.getLogger("stamp")

Rows = npt.NDArray[np.int64]

Sampler = Callable[[Rows, int, np.random.Generator], Rows]
"""(one slide's rows in raster order, how many to take, rng) -> the rows to keep"""


def first_n(rows: Rows, n: int, rng: np.random.Generator) -> Rows:
    return rows[:n]


def uniform_random(rows: Rows, n: int, rng: np.random.Generator) -> Rows:
    return np.sort(rng.choice(rows, size=n, replace=False))


SAMPLERS: dict[SamplingStrategy, Sampler] = {
    "first_n": first_n,
    "random": uniform_random,
}


def run(
    patient: Patient,
    inputs: list[Path],
    output: Path,
    *,
    settings: SamplingConfig,
    overview_dir: Path | None = None,
    overview_max_dimension: int = 4000,
) -> None:
    """Step entry point. `inputs` holds the patient's full bag.

    With `overview_dir`, also saves one image of all the patient's slides showing
    the sampled tiles, as `<overview_dir>/<patient>.png`.
    """
    (full_bag_file,) = inputs
    full = read_bag(full_bag_file)
    rng = patient_rng(settings.seed, patient.group_key)
    sampled = sample_bag(full, settings, rng)
    write_bag(output, sampled)

    if overview_dir is not None:
        _save_overview(
            full,
            sampled,
            patient,
            overview_dir / f"{patient.file_stem}.png",
            overview_max_dimension,
        )


def sample_bag(bag: Bag, settings: SamplingConfig, rng: np.random.Generator) -> Bag:
    rows_per_slide = [
        np.flatnonzero(bag.mask & (bag.slide_idx == k)) for k in range(len(bag.slides))
    ]
    available = np.array([len(rows) for rows in rows_per_slide], dtype=np.int64)
    quota = allocate(available, settings.n_tiles, redistribute=settings.redistribute)

    sampler = SAMPLERS[settings.strategy]
    chosen = np.concatenate(
        [
            sampler(rows, int(n), rng)
            for rows, n in zip(rows_per_slide, quota, strict=True)
            if n > 0
        ]
        or [np.zeros(0, np.int64)]
    )
    sampled = _pad(bag, chosen, settings)

    summary = SamplingSummary(
        settings=settings.model_dump(),
        n_available=int(available.sum()),
        n_unique=len(chosen),
        per_slide=quota.tolist(),
    )
    return dataclasses.replace(sampled, sampling=summary)


def allocate(available: npt.ArrayLike, budget: int, *, redistribute: bool) -> Rows:
    """How many tiles to take from each slide.

    Every slide that has tiles gets budget // n_slides (the remainder goes to the
    first slides), capped at the tiles it has. With `redistribute`, quota a slide
    can't use is shared out among the slides that still have tiles left, until
    the budget is spent or every tile is taken.
    """
    available = np.asarray(available, dtype=np.int64)
    taken = np.zeros_like(available)
    open_slides = np.flatnonzero(available > 0)
    remaining = budget

    while remaining > 0 and len(open_slides) > 0:
        share = _split_evenly(remaining, len(open_slides))
        taken[open_slides] += np.minimum(
            share, available[open_slides] - taken[open_slides]
        )
        remaining = budget - int(taken.sum())
        open_slides = np.flatnonzero(taken < available)
        if not redistribute:
            break
    return taken


def patient_rng(seed: int, group_key: str) -> np.random.Generator:
    """Per-patient generator, so results don't depend on processing order or machine."""
    return np.random.default_rng([seed, zlib.crc32(group_key.encode())])


def _split_evenly(total: int, parts: int) -> Rows:
    share = np.full(parts, total // parts, dtype=np.int64)
    share[: total % parts] += 1
    return share


def _save_overview(
    full: Bag, sampled: Bag, patient: Patient, output: Path, max_dimension: int
) -> None:
    """Best effort: a failed image is logged, never fails the step."""
    try:
        kept = [
            full.coords_slide[full.mask & (full.slide_idx == k)]
            for k in range(len(full.slides))
        ]
        # pad_mode repeat repeats rows: draw each tile once
        chosen = [
            np.unique(
                sampled.coords_slide[sampled.mask & (sampled.slide_idx == k)], axis=0
            )
            for k in range(len(full.slides))
        ]
        n_sampled = sum(len(c) for c in chosen)
        n_kept = sum(len(k) for k in kept)
        save_patient_overview(
            output,
            slides=full.slides,
            kept_um=kept,
            sampled_um=chosen,
            tile_size_um=full.tile_size_um,
            title=(
                f"{patient.group_key} (label {patient.label}): "
                f"{n_sampled} of {n_kept} tiles sampled"
            ),
            max_dimension=max_dimension,
        )
    except Exception:
        _logger.exception(f"failed to save sampling overview for {patient.group_key}")


def _pad(bag: Bag, chosen: Rows, settings: SamplingConfig) -> Bag:
    """The chosen rows, padded up to n_tiles if pad_mode asks for it."""
    shortfall = settings.n_tiles - len(chosen)
    if shortfall <= 0 or settings.pad_mode == "none" or len(chosen) == 0:
        return bag.select(chosen)
    if settings.pad_mode == "repeat":
        return bag.select(np.resize(chosen, settings.n_tiles))
    return bag.select(chosen).with_zero_padding(shortfall)
