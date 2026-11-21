"""Step `tile`: WSI -> per-slide bag of tile positions.

Runs STAMP's own tiler (background rejection included) and keeps its tile cache,
so later steps read tile images from the cache instead of reopening the WSI.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from stamp.distributed.fs import ensure_dir
from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.data.bags import Bag, SlideInfo, write_bag
from stamp.preprocessing.data.clinical import Slide
from stamp.preprocessing.slides.geometry import select_slide_tiles
from stamp.preprocessing.slides.overview import save_tile_overview
from stamp.preprocessing.slides.tile_cache import (
    MAX_SUPERTILE_SIZE_SLIDE_PX,
    cache_file_for,
    read_tile_positions,
)
from stamp.preprocessing.slides.wsi import read_wsi_metadata
from stamp.preprocessing.tiling import tiles_with_cache
from stamp.types import ImageExtension, Microns, SlideMPP, TilePixels

_logger = logging.getLogger("stamp")


@dataclass(frozen=True)
class TilingSettings:
    cache_dir: Path
    cache_tiles_ext: ImageExtension
    tile_size_um: Microns
    tile_size_px: TilePixels
    brightness_cutoff: int | None
    canny_cutoff: float | None
    max_workers: int
    default_slide_mpp: SlideMPP | None
    drop_edge_tiles: bool
    overview_dir: Path | None
    """Where to save a thumbnail per slide with its kept tiles (None: don't)."""
    overview_max_dimension: int

    @classmethod
    def from_config(
        cls,
        pre: PreprocessingConfig,
        *,
        cache_dir: Path,
        drop_edge_tiles: bool,
        overview_dir: Path | None,
    ) -> "TilingSettings":
        return cls(
            cache_dir=cache_dir,
            cache_tiles_ext=pre.cache_tiles_ext,
            tile_size_um=pre.tile_size_um,
            tile_size_px=pre.tile_size_px,
            brightness_cutoff=pre.brightness_cutoff,
            canny_cutoff=pre.canny_cutoff,
            max_workers=pre.max_workers,
            default_slide_mpp=pre.default_slide_mpp,
            drop_edge_tiles=drop_edge_tiles,
            overview_dir=overview_dir if pre.save_visualisations else None,
            overview_max_dimension=pre.visualisation_max_dimension,
        )


def run(
    slide: Slide, inputs: list[Path], output: Path, *, settings: TilingSettings
) -> None:
    """Step entry point. `inputs` is empty: the WSI itself is the input."""
    if slide.path is None:
        raise FileNotFoundError(f"no WSI found for slide {slide.slide_id}")

    wsi = read_wsi_metadata(slide.path, default_mpp=settings.default_slide_mpp)
    cache_file = ensure_tile_cache(slide.path, settings)

    positions = read_tile_positions(cache_file)
    rows = select_slide_tiles(
        positions,
        tile_size_um=settings.tile_size_um,
        wsi=wsi,
        drop_edge_tiles=settings.drop_edge_tiles,
    )
    if len(rows) == 0:
        _logger.warning(f"no tissue tiles found in {slide.path}")

    slide_info = SlideInfo(
        slide_id=slide.slide_id,
        slide_path=str(slide.path),
        mpp=wsi.mpp,
        width_px=wsi.width_px,
        height_px=wsi.height_px,
        cache_file=str(cache_file),
        n_tiles=len(rows),
        n_edge_tiles_dropped=len(positions) - len(rows),
    )
    bag = Bag.for_slide(
        slide_info,
        positions[rows],
        tile_size_um=settings.tile_size_um,
        tile_size_px=settings.tile_size_px,
    )
    write_bag(output, bag)

    if settings.overview_dir is not None:
        overview = settings.overview_dir / f"{slide.file_stem}.png"
        _save_overview(overview, bag, settings)


def ensure_tile_cache(slide_path: Path, settings: TilingSettings) -> Path:
    """Tile the slide into STAMP's cache unless a cache for these settings exists."""
    cache_file = cache_file_for(
        slide_path,
        cache_dir=settings.cache_dir,
        cache_tiles_ext=settings.cache_tiles_ext,
        tile_size_um=settings.tile_size_um,
        tile_size_px=settings.tile_size_px,
        brightness_cutoff=settings.brightness_cutoff,
        canny_cutoff=settings.canny_cutoff,
    )
    if cache_file.exists():
        return cache_file

    ensure_dir(settings.cache_dir)
    tiles = tiles_with_cache(
        slide_path,
        cache_dir=settings.cache_dir,
        cache_tiles_ext=settings.cache_tiles_ext,
        tile_size_um=settings.tile_size_um,
        tile_size_px=settings.tile_size_px,
        max_supertile_size_slide_px=MAX_SUPERTILE_SIZE_SLIDE_PX,
        max_workers=settings.max_workers,
        brightness_cutoff=settings.brightness_cutoff,
        canny_cutoff=settings.canny_cutoff,
        default_slide_mpp=settings.default_slide_mpp,
    )
    for _ in tiles:  # writing the cache is a side effect of iterating
        pass

    if not cache_file.exists():
        raise RuntimeError(
            f"STAMP's tiler did not create the expected cache {cache_file}"
        )
    return cache_file


def _save_overview(output: Path, bag: Bag, settings: TilingSettings) -> None:
    """Best effort: a thumbnail with the kept tiles in red. Never fails the step."""
    slide = bag.slides[0]
    try:
        save_tile_overview(
            output,
            slide=slide,
            tile_size_um=bag.tile_size_um,
            highlighted_um=bag.coords_slide,
            caption=f"{slide.slide_id}: {len(bag)} tiles kept",
            max_dimension=settings.overview_max_dimension,
        )
    except Exception:
        _logger.exception(f"failed to save tile overview for {slide.slide_id}")
