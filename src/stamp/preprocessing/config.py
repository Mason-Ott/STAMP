from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stamp.types import ImageExtension, Microns, SlideMPP, TilePixels

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


class ExtractorName(StrEnum):
    CTRANSPATH = "ctranspath"
    CHIEF_CTRANSPATH = "chief-ctranspath"
    CONCH = "conch"
    CONCH1_5 = "conch1_5"
    UNI = "uni"
    UNI2 = "uni2"
    GIGAPATH = "gigapath"
    VIRCHOW = "virchow"
    VIRCHOW_FULL = "virchow-full"
    VIRCHOW2 = "virchow2"
    RESNET50 = "resnet50"


class PreprocessingConfig(BaseModel, arbitrary_types_allowed=True):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    wsi_dir: Path
    wsi_list: Path | None = Field(
        default=None, description="Txt, Excel or CSV to read data filename from"
    )
    cache_dir: Path | None = None
    cache_tiles_ext: ImageExtension = "jpg"
    tile_size_um: Microns = Microns(256.0)
    tile_size_px: TilePixels = TilePixels(224)
    extractor: ExtractorName
    max_workers: int = 8
    device: str = Field(
        default_factory=lambda: (
            "cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
    )
    generate_hash: bool = True
    
    save_visualisations: bool = Field(
        default=False,
        description="Save WSI overview images showing successfully extracted tiles.",
    )
    visualisation_max_dimension: int = Field(
        default=4000,
        gt=0,
        description="Maximum width or height of generated overview images in pixels.",
    )

    default_slide_mpp: SlideMPP | None = None
    """MPP of the slide to use if none can be inferred from the WSI"""

    # Background rejection
    brightness_cutoff: int | None = Field(240, gt=0, lt=255)
    """Any tile brighter than this will be discarded as probable background.
    If set to `None`, the brightness-based background rejection is disabled.
    """

    canny_cutoff: float | None = Field(0.02, gt=0.0, lt=1.0)
    """Any tile with a lower ratio of pixels classified as "edges" than this
    will be rejected.
    If set to `None`, brightness-based rejection is disabled.
    """
