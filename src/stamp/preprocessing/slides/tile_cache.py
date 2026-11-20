"""Access to STAMP's tile cache (one zip of tile images per slide).

Everything the pipeline knows about the cache's format lives here. The cache is
written by `stamp.preprocessing.tiling.tiles_with_cache`, unchanged, so it is
shared with plain `stamp preprocess`.
"""

import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
from PIL import Image

from stamp.preprocessing import tiling
from stamp.types import ImageExtension, Microns, SlidePixels, TilePixels

MAX_SUPERTILE_SIZE_SLIDE_PX = SlidePixels(2**10)
"""The supertile size `stamp preprocess` passes to the tiler. It is part of the
cache key, so it must match for the two to share a cache."""

PARAMS_MEMBER = "tiler_params.json"

TilePosition = tuple[float, float]
"""(x_um, y_um) of a tile's top-left corner on its slide."""


def cache_file_for(
    slide_path: Path,
    *,
    cache_dir: Path,
    cache_tiles_ext: ImageExtension,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    brightness_cutoff: int | None,
    canny_cutoff: float | None,
) -> Path:
    """The cache zip `tiles_with_cache` reads or writes for these settings."""
    params = tiling.make_tiler_params(
        slide_path,
        cache_tiles_ext=cache_tiles_ext,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        max_supertile_size_slide_px=MAX_SUPERTILE_SIZE_SLIDE_PX,
        brightness_cutoff=brightness_cutoff,
        canny_cutoff=canny_cutoff,
    )
    return tiling.cache_file_for(cache_dir, slide_path, params)


def tile_members(cache: ZipFile) -> dict[TilePosition, str]:
    """Map tile position -> zip member name for every tile in a cache zip."""
    params = json.loads(cache.read(PARAMS_MEMBER).decode())
    ext = re.escape(params.get("tile_ext", "jpg"))
    pattern = re.compile(rf"tile_\((\d+\.\d+), (\d+\.\d+)\)\.{ext}")
    members = {}
    for name in cache.namelist():
        if match := pattern.fullmatch(name):
            members[(float(match.group(1)), float(match.group(2)))] = name
    return members


def read_tile_positions(cache_file: Path) -> npt.NDArray[np.float64]:
    """(N, 2) positions in um of every tile in a cache zip, in no particular order."""
    with ZipFile(cache_file) as cache:
        positions = list(tile_members(cache))
    return np.array(positions, dtype=np.float64).reshape(-1, 2)


class TileReader:
    """Reads tile images from cache zips, keeping each zip open between reads.

    Not shared between processes: create one per DataLoader worker (opening is
    lazy, so a reader created before the workers fork starts with no open files).
    """

    def __init__(self) -> None:
        self._open: dict[str, tuple[ZipFile, dict[TilePosition, str]]] = {}

    def read(self, cache_file: str, position: TilePosition) -> Image.Image:
        if cache_file not in self._open:
            cache = ZipFile(cache_file)
            self._open[cache_file] = (cache, tile_members(cache))
        cache, members = self._open[cache_file]
        try:
            member = members[position]
        except KeyError:
            raise KeyError(f"no tile at {position} in {cache_file}") from None
        with cache.open(member) as fp:
            image = Image.open(fp)
            image.load()
        return image

    def close(self) -> None:
        for cache, _ in self._open.values():
            cache.close()
        self._open.clear()
