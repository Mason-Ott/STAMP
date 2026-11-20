"""Step `import_slides`: features from plain `stamp preprocess` -> per-slide bag.

Lets the `extract_then_sample` order reuse features that were already extracted,
instead of tiling and extracting again. The imported bags follow the same rules
as freshly made ones (raster order, edge tiles dropped), so the patient bags
built from them select the same tiles.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from stamp.preprocessing.data.bags import Bag, SlideInfo, write_bag
from stamp.preprocessing.data.clinical import Slide
from stamp.preprocessing.slides.geometry import select_slide_tiles
from stamp.preprocessing.slides.wsi import read_wsi_metadata
from stamp.types import SlideMPP


@dataclass(frozen=True)
class ImportSettings:
    feat_dir: Path
    wsi_dir: Path
    default_slide_mpp: SlideMPP | None
    drop_edge_tiles: bool


def stamp_feature_file(slide: Slide, settings: ImportSettings) -> Path:
    """Where `stamp preprocess` put this slide's features.

    It mirrors the WSI folder structure under feat_dir; for slides directly in
    wsi_dir that is simply `<feat_dir>/<slide_id>.h5`.
    """
    if slide.path is not None and slide.path.is_relative_to(settings.wsi_dir):
        return settings.feat_dir / slide.path.relative_to(settings.wsi_dir).with_suffix(
            ".h5"
        )
    return settings.feat_dir / f"{slide.slide_id}.h5"


def run(
    slide: Slide, inputs: list[Path], output: Path, *, settings: ImportSettings
) -> None:
    """Step entry point. `inputs` holds the slide's STAMP feature file."""
    (feature_file,) = inputs
    if slide.path is None:
        raise FileNotFoundError(f"no WSI found for slide {slide.slide_id}")

    with h5py.File(feature_file, "r") as f:
        coords = f["coords"][()].astype(np.float64)
        feats = f["feats"][()]
        tile_size_um, tile_size_px = _tile_size(f.attrs, feature_file)
        extractor = f.attrs.get("extractor")

    wsi = read_wsi_metadata(slide.path, default_mpp=settings.default_slide_mpp)
    rows = select_slide_tiles(
        coords,
        tile_size_um=tile_size_um,
        wsi=wsi,
        drop_edge_tiles=settings.drop_edge_tiles,
    )
    slide_info = SlideInfo(
        slide_id=slide.slide_id,
        slide_path=str(slide.path),
        mpp=wsi.mpp,
        width_px=wsi.width_px,
        height_px=wsi.height_px,
        cache_file=None,
        n_tiles=len(rows),
        n_edge_tiles_dropped=len(coords) - len(rows),
    )
    bag = Bag.for_slide(
        slide_info,
        coords[rows],
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        feats=feats[rows],
        extractor=None if extractor is None else str(extractor),
    )
    write_bag(output, bag)


def _tile_size(attrs: h5py.AttributeManager, path: Path) -> tuple[float, int]:
    """Tile size from a STAMP feature file's attributes (current and older names)."""
    tile_size_um = attrs.get("tile_size_um")
    if tile_size_um is None and attrs.get("unit") == "um":
        tile_size_um = attrs.get("tile_size")
    tile_size_px = attrs.get("tile_size_px")
    if tile_size_um is None or tile_size_px is None:
        raise ValueError(
            f"{path} doesn't record its tile size; "
            "re-extract it with this STAMP version"
        )
    return float(tile_size_um), int(tile_size_px)
