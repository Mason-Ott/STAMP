"""Slide thumbnails with tiles drawn on top, to check tiling and sampling by eye.

Colours: green tiles are used, red tiles were kept by tiling but not sampled.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import openslide
from PIL import Image, ImageDraw

from stamp.preprocessing.data.bags import SlideInfo

USED_STYLE = {"fill": (0, 170, 0, 70), "outline": (0, 140, 0, 220)}
"""Green: tiles that are used (kept by tiling, or sampled)."""

UNUSED_STYLE = {"fill": (230, 0, 0, 70), "outline": (200, 0, 0, 220)}
"""Red: tiles kept by tiling but dropped by sampling."""

LABEL_HEIGHT = 22
TITLE_HEIGHT = 28
MAX_COLUMNS = 3


@dataclass(frozen=True)
class SlidePanel:
    """One slide of a patient overview."""

    thumbnail: Image.Image
    """The whole slide."""

    slide: SlideInfo
    kept_um: npt.NDArray[np.floating]
    """Top-left corners (um, on this slide) of the tiles tiling kept."""

    sampled_um: npt.NDArray[np.floating]
    """Subset of `kept_um` that sampling took."""

    label: str


def save_tile_overview(
    output: Path,
    *,
    slide: SlideInfo,
    tile_size_um: float,
    highlighted_um: npt.NDArray[np.floating],
    caption: str,
    max_dimension: int,
) -> None:
    """Save a thumbnail of `slide` with the `highlighted_um` tiles in green.
    Coordinates are tile top-left corners in um."""
    thumbnail = read_thumbnail(slide, max_dimension).convert("RGBA")
    overlay = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = np.array(
        [thumbnail.width / slide.width_px, thumbnail.height / slide.height_px]
    )
    tile_size = tile_size_um / slide.mpp * scale
    line_width = max(1, round(max(thumbnail.size) / 2000))
    for top_left in highlighted_um / slide.mpp * scale:
        draw.rectangle(
            (*top_left, *(top_left + tile_size)), width=line_width, **USED_STYLE
        )

    draw.rectangle((0, 0, 7 * len(caption) + 12, 22), fill=(255, 255, 255, 200))
    draw.text((6, 5), caption, fill=(0, 0, 0, 255))

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(thumbnail, overlay).convert("RGB").save(output)


def save_patient_overview(
    output: Path,
    *,
    slides: Sequence[SlideInfo],
    kept_um: Sequence[npt.NDArray[np.floating]],
    sampled_um: Sequence[npt.NDArray[np.floating]],
    tile_size_um: float,
    title: str,
    max_dimension: int,
) -> None:
    """Save one image with all of a patient's slides, tissue only, at a common
    scale. Reads the thumbnails, then hands over to `draw_patient_overview`."""
    panels = [
        SlidePanel(
            thumbnail=read_thumbnail(slide, max_dimension),
            slide=slide,
            kept_um=kept,
            sampled_um=sampled,
            label=f"{slide.slide_id}: {len(sampled)} of {len(kept)} tiles sampled",
        )
        for slide, kept, sampled in zip(slides, kept_um, sampled_um, strict=True)
    ]
    image = draw_patient_overview(
        panels, tile_size_um=tile_size_um, title=title, max_dimension=max_dimension
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def read_thumbnail(slide: SlideInfo, max_dimension: int) -> Image.Image:
    wsi = openslide.open_slide(slide.slide_path)
    try:
        return wsi.get_thumbnail((max_dimension, max_dimension)).convert("RGB")
    finally:
        wsi.close()


def draw_patient_overview(
    panels: Sequence[SlidePanel],
    *,
    tile_size_um: float,
    title: str,
    max_dimension: int,
    max_columns: int = MAX_COLUMNS,
) -> Image.Image:
    """Slides side by side (in a grid past `max_columns`), each cropped to its
    tissue, all at the same um-per-pixel so tiles are the same size everywhere.

    Pure drawing: no files are read. The longest side is about `max_dimension`
    (plus the label and title bars).
    """
    if not panels:
        raise ValueError("no slides to draw")
    gap_um = tile_size_um
    crops = [_tissue_box_um(panel, tile_size_um) for panel in panels]
    sizes_um = [(x1 - x0, y1 - y0) for x0, y0, x1, y1 in crops]

    n_columns = min(len(panels), max_columns)
    rows = [
        list(range(i, min(i + n_columns, len(panels))))
        for i in range(0, len(panels), n_columns)
    ]
    total_w_um = max(
        sum(sizes_um[i][0] for i in row) + gap_um * (len(row) - 1) for row in rows
    )
    total_h_um = sum(max(sizes_um[i][1] for i in row) for row in rows) + gap_um * (
        len(rows) - 1
    )

    px_per_um = min(max_dimension / total_w_um, max_dimension / total_h_um)
    for panel in panels:  # never magnify a thumbnail beyond its own resolution
        px_per_um = min(
            px_per_um, panel.thumbnail.width / (panel.slide.width_px * panel.slide.mpp)
        )
    gap = max(2, round(gap_um * px_per_um))

    rendered = {
        i: _render_panel(panel, crop, tile_size_um, px_per_um)
        for i, (panel, crop) in enumerate(zip(panels, crops, strict=True))
    }
    row_widths = [
        sum(rendered[i].width for i in row) + gap * (len(row) - 1) for row in rows
    ]
    row_heights = [max(rendered[i].height for i in row) + LABEL_HEIGHT for row in rows]
    width = max(max(row_widths), 420)
    height = TITLE_HEIGHT + sum(row_heights) + gap * (len(rows) - 1)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(0, 0, 0))
    _draw_legend(draw, x=6 + 7 * len(title) + 24, y=7)

    y = TITLE_HEIGHT
    for row, row_height in zip(rows, row_heights, strict=True):
        x = 0
        for i in row:
            draw.text((x + 4, y + 5), panels[i].label, fill=(0, 0, 0))
            canvas.paste(rendered[i], (x, y + LABEL_HEIGHT))
            x += rendered[i].width + gap
        y += row_height + gap
    return canvas


def _tissue_box_um(panel: SlidePanel, tile_size_um: float) -> tuple[float, ...]:
    """Bounding box of the kept tiles plus one tile of margin, clipped to the slide."""
    slide = panel.slide
    full_w = slide.width_px * slide.mpp
    full_h = slide.height_px * slide.mpp
    if len(panel.kept_um) == 0:
        return 0.0, 0.0, min(full_w, 4 * tile_size_um), min(full_h, 4 * tile_size_um)
    low = panel.kept_um.min(axis=0) - tile_size_um
    high = panel.kept_um.max(axis=0) + 2 * tile_size_um
    return (
        max(0.0, float(low[0])),
        max(0.0, float(low[1])),
        min(full_w, float(high[0])),
        min(full_h, float(high[1])),
    )


def _render_panel(
    panel: SlidePanel,
    crop_um: tuple[float, ...],
    tile_size_um: float,
    px_per_um: float,
) -> Image.Image:
    x0, y0, x1, y1 = crop_um
    slide = panel.slide
    thumb_scale = np.array(
        [
            panel.thumbnail.width / (slide.width_px * slide.mpp),
            panel.thumbnail.height / (slide.height_px * slide.mpp),
        ]
    )
    box = (
        x0 * thumb_scale[0],
        y0 * thumb_scale[1],
        x1 * thumb_scale[0],
        y1 * thumb_scale[1],
    )
    size = (max(1, round((x1 - x0) * px_per_um)), max(1, round((y1 - y0) * px_per_um)))
    image = panel.thumbnail.resize(size, Image.Resampling.LANCZOS, box=box).convert(
        "RGBA"
    )

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    tile_px = tile_size_um * px_per_um
    line_width = max(1, round(max(image.size) / 1500))
    origin = np.array([x0, y0])

    sampled = {tuple(row) for row in panel.sampled_um.tolist()}
    is_sampled = np.array(
        [tuple(row) in sampled for row in panel.kept_um.tolist()], dtype=bool
    )
    for tiles, style in (
        (panel.kept_um[~is_sampled], UNUSED_STYLE),
        (panel.kept_um[is_sampled], USED_STYLE),
    ):
        for top_left in (tiles - origin) * px_per_um:
            draw.rectangle(
                (*top_left, *(top_left + tile_px)), width=line_width, **style
            )
    return Image.alpha_composite(image, overlay).convert("RGB")


def _draw_legend(draw: ImageDraw.ImageDraw, *, x: int, y: int) -> None:
    for text, style in (("sampled", USED_STYLE), ("not sampled", UNUSED_STYLE)):
        draw.rectangle(
            (x, y, x + 12, y + 12),
            fill=style["outline"][:3],
        )
        draw.text((x + 18, y + 1), text, fill=(0, 0, 0))
        x += 18 + 7 * len(text) + 16
