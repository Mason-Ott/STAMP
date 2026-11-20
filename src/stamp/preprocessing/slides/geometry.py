"""Tile ordering and position checks (all coordinates in micrometres)."""

import numpy as np
import numpy.typing as npt

from stamp.preprocessing.slides.wsi import WsiMetadata

EDGE_TOLERANCE_PX = 1.0
"""Slack for rounding when converting between micrometres and pixels."""


def select_slide_tiles(
    coords_um: npt.NDArray[np.floating],
    *,
    tile_size_um: float,
    wsi: WsiMetadata,
    drop_edge_tiles: bool,
) -> npt.NDArray[np.int64]:
    """Rows of a slide's tiles to keep, in raster order.

    Used by every step that creates a per-slide bag, so all per-slide bags follow
    the same rules no matter where their tiles came from.
    """
    rows = raster_order(coords_um)
    if drop_edge_tiles:
        rows = rows[
            tiles_inside_slide(coords_um[rows], tile_size_um=tile_size_um, wsi=wsi)
        ]
    return rows


def raster_order(coords_um: npt.NDArray[np.floating]) -> npt.NDArray[np.int64]:
    """Indices that sort tiles top-to-bottom, then left-to-right.

    STAMP's tiler yields tiles in thread-completion order, which varies between
    runs. Sorting makes "the first N tiles" reproducible.
    """
    return np.lexsort((coords_um[:, 0], coords_um[:, 1]))


def tiles_inside_slide(
    coords_um: npt.NDArray[np.floating], *, tile_size_um: float, wsi: WsiMetadata
) -> npt.NDArray[np.bool_]:
    """True for tiles that lie entirely inside the scanned area.

    STAMP's tiler reads whole supertiles, so near the bottom/right border it can
    produce tiles that extend past the edge of the scan, where the slide reader
    returns black. The black/background boundary passes STAMP's edge-density
    filter, so those tiles survive background rejection unless removed here.
    """
    far_corner_px = (coords_um + tile_size_um) / wsi.mpp
    return (far_corner_px[:, 0] <= wsi.width_px + EDGE_TOLERANCE_PX) & (
        far_corner_px[:, 1] <= wsi.height_px + EDGE_TOLERANCE_PX
    )
