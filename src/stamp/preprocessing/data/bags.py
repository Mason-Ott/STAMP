"""The tile bag: the one data structure every step reads and writes.

A bag is a set of tiles from one slide or one patient, with or without features,
stored as one HDF5 file:

datasets
  coords        (N, 2) float64  um. Patient bags: *virtual* coordinates, with the
                patient's slides laid side by side along x. Slide bags: same as
                coords_slide. Named `coords` so STAMP (training, encode_slides,
                TITAN) can read a patient bag as if it were one slide.
  coords_slide  (N, 2) float64  um, top-left corner of the tile on its own WSI
  slide_idx     (N,)   int32    index into `slides` (-1 on padding rows)
  tile_idx      (N,)   int64    row of the tile in its per-slide bag (-1 on padding)
  mask          (N,)   bool     False on zero-padding rows
  feats         (N, D)          only once features have been extracted

attributes
  tile_size_um, tile_size_px, unit ("um"), feat_type ("tile" with features,
  "positions" without), n_valid, pipeline_format, stamp_version,
  slides (JSON list of SlideInfo), extractor, sampling (JSON SamplingSummary),
  group_key, patient_id, hospital, label, missing_slide_ids (JSON)

Invariant: within each slide, rows are in raster order (top-to-bottom, then
left-to-right). Every step preserves it.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from stamp.preprocessing.data.files import atomic_h5_file

PIPELINE_FORMAT_VERSION = 2

PADDING_INDEX = -1


@dataclass(frozen=True)
class SlideInfo:
    """Where a slide's tiles came from and where they sit in the bag's coords."""

    slide_id: str
    slide_path: str
    mpp: float
    width_px: int
    height_px: int
    cache_file: str | None
    """STAMP tile cache zip holding the tile images (None for imported features)."""

    n_tiles: int
    """Tiles of this slide in the bag it was first written to (before sampling)."""

    n_edge_tiles_dropped: int = 0
    x_offset_um: float = 0.0
    y_offset_um: float = 0.0
    """coords = coords_slide + (x_offset_um, y_offset_um) for this slide's rows."""


@dataclass(frozen=True)
class PatientInfo:
    group_key: str
    patient_id: str
    hospital: str
    label: int
    missing_slide_ids: tuple[str, ...] = ()
    """Slides listed in the clinical table that weren't available when combining."""


@dataclass(frozen=True)
class SamplingSummary:
    settings: dict[str, Any]
    n_available: int
    """Real tiles in the patient bag before sampling."""
    n_unique: int
    """Distinct real tiles after sampling (excludes repeats and zero padding)."""
    per_slide: list[int]
    """Distinct tiles taken from each slide."""


@dataclass(frozen=True)
class Bag:
    coords: npt.NDArray[np.float64]
    coords_slide: npt.NDArray[np.float64]
    slide_idx: npt.NDArray[np.int32]
    tile_idx: npt.NDArray[np.int64]
    mask: npt.NDArray[np.bool_]
    tile_size_um: float
    tile_size_px: int
    slides: tuple[SlideInfo, ...]
    patient: PatientInfo | None = None
    feats: npt.NDArray[np.floating] | None = None
    extractor: str | None = None
    sampling: SamplingSummary | None = None

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def for_slide(
        cls,
        slide: SlideInfo,
        coords_um: npt.NDArray[np.floating],
        *,
        tile_size_um: float,
        tile_size_px: int,
        feats: npt.NDArray[np.floating] | None = None,
        extractor: str | None = None,
    ) -> "Bag":
        """A single-slide bag. `coords_um` must already be in raster order."""
        n = len(coords_um)
        coords = np.asarray(coords_um, dtype=np.float64).reshape(n, 2)
        return cls(
            coords=coords,
            coords_slide=coords.copy(),
            slide_idx=np.zeros(n, np.int32),
            tile_idx=np.arange(n, dtype=np.int64),
            mask=np.ones(n, bool),
            tile_size_um=float(tile_size_um),
            tile_size_px=int(tile_size_px),
            slides=(slide,),
            feats=feats,
            extractor=extractor,
        )

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def has_feats(self) -> bool:
        return self.feats is not None

    def select(self, rows: npt.NDArray[np.integer]) -> "Bag":
        """A bag with only the given rows, in the given order (repeats allowed)."""
        return dataclasses.replace(
            self,
            coords=self.coords[rows],
            coords_slide=self.coords_slide[rows],
            slide_idx=self.slide_idx[rows],
            tile_idx=self.tile_idx[rows],
            mask=self.mask[rows],
            feats=None if self.feats is None else self.feats[rows],
        )

    def with_zero_padding(self, n_rows: int) -> "Bag":
        """Append `n_rows` padding rows (mask False, zero coordinates and features)."""
        feats = self.feats
        if feats is not None:
            feats = np.concatenate(
                [feats, np.zeros((n_rows, feats.shape[1]), feats.dtype)]
            )
        return dataclasses.replace(
            self,
            coords=np.concatenate([self.coords, np.zeros((n_rows, 2))]),
            coords_slide=np.concatenate([self.coords_slide, np.zeros((n_rows, 2))]),
            slide_idx=np.concatenate(
                [self.slide_idx, np.full(n_rows, PADDING_INDEX, np.int32)]
            ),
            tile_idx=np.concatenate(
                [self.tile_idx, np.full(n_rows, PADDING_INDEX, np.int64)]
            ),
            mask=np.concatenate([self.mask, np.zeros(n_rows, bool)]),
            feats=feats,
        )

    def distinct_tiles(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Collapse repeats so each real tile is processed once.

        Returns
            rows:    one row per distinct real tile (its first occurrence), in bag order
            lookup:  for every real row (mask True, in bag order), the position in
                     `rows` of the same tile
        """
        valid = np.flatnonzero(self.mask)
        keys = np.stack([self.slide_idx[valid], self.tile_idx[valid]], axis=1)
        _, first, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        in_bag_order = np.argsort(first)
        position = np.empty_like(in_bag_order)
        position[in_bag_order] = np.arange(len(in_bag_order))
        return valid[first[in_bag_order]], position[inverse.reshape(-1)]

    def _validate(self) -> None:
        n = len(self.coords)
        shapes = {
            "coords": (self.coords.shape, (n, 2)),
            "coords_slide": (self.coords_slide.shape, (n, 2)),
            "slide_idx": (self.slide_idx.shape, (n,)),
            "tile_idx": (self.tile_idx.shape, (n,)),
            "mask": (self.mask.shape, (n,)),
        }
        for name, (actual, expected) in shapes.items():
            if actual != expected:
                raise ValueError(
                    f"bag field {name} has shape {actual}, expected {expected}"
                )
        if self.feats is not None and (self.feats.ndim != 2 or len(self.feats) != n):
            raise ValueError(
                f"bag feats have shape {self.feats.shape}, expected ({n}, D)"
            )
        real = self.slide_idx[self.mask]
        if len(real) and (real.min() < 0 or real.max() >= len(self.slides)):
            raise ValueError("bag slide_idx points outside its slides list")


def write_bag(path: Path, bag: Bag) -> None:
    """Write `bag` to `path` atomically (see files.atomic_h5_file)."""
    import stamp

    attrs: dict[str, Any] = {
        "tile_size_um": bag.tile_size_um,
        "tile_size_px": bag.tile_size_px,
        "unit": "um",
        "feat_type": "tile" if bag.has_feats else "positions",
        "n_valid": int(bag.mask.sum()),
        "pipeline_format": PIPELINE_FORMAT_VERSION,
        "stamp_version": stamp.__version__,
        "slides": json.dumps([dataclasses.asdict(s) for s in bag.slides]),
    }
    if bag.extractor is not None:
        attrs["extractor"] = bag.extractor
    if bag.sampling is not None:
        attrs["sampling"] = json.dumps(dataclasses.asdict(bag.sampling))
    if bag.patient is not None:
        attrs.update(patient_attrs(bag.patient))

    with atomic_h5_file(path) as f:
        f["coords"] = bag.coords
        f["coords_slide"] = bag.coords_slide
        f["slide_idx"] = bag.slide_idx
        f["tile_idx"] = bag.tile_idx
        f["mask"] = bag.mask
        if bag.feats is not None:
            f["feats"] = bag.feats
        f.attrs.update(attrs)


def read_bag(path: Path, *, with_feats: bool = True) -> Bag:
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        version = attrs.get("pipeline_format")
        if version != PIPELINE_FORMAT_VERSION:
            raise ValueError(
                f"{path} is not a pipeline bag of format {PIPELINE_FORMAT_VERSION} "
                f"(found {version!r}); re-run the step that produced it"
            )
        sampling = attrs.get("sampling")
        return Bag(
            coords=f["coords"][()],
            coords_slide=f["coords_slide"][()],
            slide_idx=f["slide_idx"][()],
            tile_idx=f["tile_idx"][()],
            mask=f["mask"][()],
            feats=f["feats"][()] if with_feats and "feats" in f else None,
            tile_size_um=float(attrs["tile_size_um"]),
            tile_size_px=int(attrs["tile_size_px"]),
            slides=tuple(SlideInfo(**s) for s in json.loads(attrs["slides"])),
            patient=_patient_from_attrs(attrs),
            extractor=_str_or_none(attrs.get("extractor")),
            sampling=SamplingSummary(**json.loads(sampling)) if sampling else None,
        )


def patient_attrs(patient: PatientInfo) -> dict[str, Any]:
    """HDF5 attributes describing a patient (also used for embedding files)."""
    return {
        "group_key": patient.group_key,
        "patient_id": patient.patient_id,
        "hospital": patient.hospital,
        "label": patient.label,
        "missing_slide_ids": json.dumps(list(patient.missing_slide_ids)),
    }


def _patient_from_attrs(attrs: dict[str, Any]) -> PatientInfo | None:
    if "group_key" not in attrs:
        return None
    return PatientInfo(
        group_key=str(attrs["group_key"]),
        patient_id=str(attrs["patient_id"]),
        hospital=str(attrs["hospital"]),
        label=int(attrs["label"]),
        missing_slide_ids=tuple(json.loads(attrs.get("missing_slide_ids", "[]"))),
    )


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)
