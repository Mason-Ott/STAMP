"""Reading slide-level metadata from a WSI."""

from dataclasses import dataclass
from pathlib import Path

import openslide

from stamp.preprocessing.tiling import MPPExtractionError, get_slide_mpp_
from stamp.types import SlideMPP


@dataclass(frozen=True)
class WsiMetadata:
    mpp: float
    """Micrometres per pixel at full resolution."""
    width_px: int
    height_px: int


def read_wsi_metadata(path: Path, *, default_mpp: SlideMPP | None) -> WsiMetadata:
    slide = openslide.open_slide(path)
    try:
        mpp = get_slide_mpp_(slide, default_mpp=default_mpp)  # type: ignore[arg-type]
        width_px, height_px = slide.dimensions
    finally:
        slide.close()
    if mpp is None:
        raise MPPExtractionError(
            f"no MPP for {path}; set preprocessing.default_slide_mpp"
        )
    return WsiMetadata(mpp=float(mpp), width_px=int(width_px), height_px=int(height_px))
