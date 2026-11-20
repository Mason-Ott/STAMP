"""Step `combine`: a patient's per-slide bags -> one patient bag.

Concatenating slide files directly breaks three things, which this step keeps:
  * distinct positions: slides are laid side by side along x (with a gap) in
    `coords`, so tiles from different slides never share a position;
  * traceability: every tile keeps `coords_slide` and `slide_idx`, so features
    can still be extracted after combining;
  * metadata: tile size, units and extractor stay, so STAMP and TITAN can read
    the result.
"""

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stamp.preprocessing.data.bags import Bag, PatientInfo, read_bag, write_bag
from stamp.preprocessing.data.clinical import Patient

_logger = logging.getLogger("stamp")


@dataclass(frozen=True)
class CombineSettings:
    gap_tiles: int
    """Empty tiles between neighbouring slides in the virtual layout."""


def run(
    patient: Patient, inputs: list[Path], output: Path, *, settings: CombineSettings
) -> None:
    """Step entry point. `inputs`: the per-slide bags that exist, in slide order."""
    bags = [read_bag(path) for path in inputs]
    found = {bag.slides[0].slide_id for bag in bags}
    missing = tuple(s.slide_id for s in patient.slides if s.slide_id not in found)
    info = PatientInfo(
        group_key=patient.group_key,
        patient_id=patient.patient_id,
        hospital=patient.hospital,
        label=patient.label,
        missing_slide_ids=missing,
    )
    write_bag(output, combine_slide_bags(bags, info, gap_tiles=settings.gap_tiles))


def combine_slide_bags(
    bags: Sequence[Bag], patient: PatientInfo, *, gap_tiles: int
) -> Bag:
    """One bag holding every tile of `bags` (each a single-slide bag), in order."""
    if not bags:
        raise ValueError(f"{patient.group_key}: no slides to combine")
    _check_compatible(bags, patient.group_key)
    if not any(len(bag) for bag in bags):
        raise ValueError(f"{patient.group_key}: none of the slides has any tiles")
    for bag in bags:
        if not len(bag):
            _logger.warning(
                f"{patient.group_key}: slide {bag.slides[0].slide_id} has no tiles"
            )

    offsets = _side_by_side_offsets(bags, gap_um=gap_tiles * bags[0].tile_size_um)

    return Bag(
        coords=np.concatenate(
            [
                bag.coords_slide + offset
                for bag, offset in zip(bags, offsets, strict=True)
            ]
        ),
        coords_slide=np.concatenate([bag.coords_slide for bag in bags]),
        slide_idx=np.concatenate(
            [np.full(len(bag), k, np.int32) for k, bag in enumerate(bags)]
        ),
        tile_idx=np.concatenate([bag.tile_idx for bag in bags]),
        mask=np.concatenate([bag.mask for bag in bags]),
        feats=_concat_feats(bags),
        tile_size_um=bags[0].tile_size_um,
        tile_size_px=bags[0].tile_size_px,
        slides=tuple(
            dataclasses.replace(
                bag.slides[0],
                x_offset_um=float(offset[0]),
                y_offset_um=float(offset[1]),
            )
            for bag, offset in zip(bags, offsets, strict=True)
        ),
        patient=patient,
        extractor=bags[0].extractor,
    )


def _concat_feats(bags: Sequence[Bag]) -> np.ndarray | None:
    feats = [bag.feats for bag in bags if bag.feats is not None]
    return np.concatenate(feats) if feats else None


def _check_compatible(bags: Sequence[Bag], group_key: str) -> None:
    first = bags[0]
    for bag in bags:
        slide_id = bag.slides[0].slide_id if bag.slides else "?"
        if len(bag.slides) != 1:
            raise ValueError(
                f"{group_key}: expected single-slide bags, got {len(bag.slides)}"
            )
        if not np.isclose(bag.tile_size_um, first.tile_size_um) or (
            bag.tile_size_px != first.tile_size_px
        ):
            raise ValueError(
                f"{group_key}: slide {slide_id} has tiles of "
                f"{bag.tile_size_um}um/{bag.tile_size_px}px, expected "
                f"{first.tile_size_um}um/{first.tile_size_px}px"
            )
        if bag.has_feats != first.has_feats or bag.extractor != first.extractor:
            raise ValueError(f"{group_key}: slide {slide_id} has different features")
        if (
            bag.feats is not None
            and first.feats is not None
            and bag.feats.shape[1] != first.feats.shape[1]
        ):
            raise ValueError(
                f"{group_key}: slide {slide_id} has a different feature size"
            )


def _side_by_side_offsets(bags: Sequence[Bag], *, gap_um: float) -> list[np.ndarray]:
    """Shift (x, y) in um that moves each slide into its own region of the patient's
    virtual coordinate space: coords = coords_slide + offset.

    Slide k's top-left tile lands at (x_start_k, 0), where x_start_k is just past
    the right edge of slide k-1 plus the gap. Empty slides take up no space.
    """
    offsets = []
    x_start = 0.0
    for bag in bags:
        if len(bag) == 0:
            offsets.append(np.zeros(2))
            continue
        top_left = bag.coords_slide.min(axis=0)
        offsets.append(np.array([x_start - top_left[0], -top_left[1]]))
        width = bag.coords_slide[:, 0].max() - top_left[0] + bag.tile_size_um
        x_start += width + gap_um
    return offsets
