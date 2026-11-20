"""Steps `extract` and `extract_slides`: bag of tile positions -> bag with features.

The same code serves both: `extract_slides` runs on per-slide bags,
`extract` on (sampled) patient bags. Tile images are read from STAMP's tile
cache, written by the `tile` step. Repeated tiles (pad_mode: repeat) are only
run through the model once, and zero-padding rows get all-zero features.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from stamp.preprocessing.data.bags import Bag, read_bag, write_bag
from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.slides.tile_cache import TilePosition, TileReader
from stamp.preprocessing.steps.models import Lazy


@dataclass(frozen=True)
class ExtractionSettings:
    device: str
    batch_size: int
    num_workers: int


def run(
    item: object,
    inputs: list[Path],
    output: Path,
    *,
    extractor: Lazy[Extractor],
    settings: ExtractionSettings,
) -> None:
    """Step entry point. `inputs` holds the bag of tile positions; `item` is unused."""
    (positions_file,) = inputs
    bag = read_bag(positions_file, with_feats=False)
    write_bag(output, extract_features(bag, extractor.get(), settings))


def extract_features(
    bag: Bag, extractor: Extractor, settings: ExtractionSettings
) -> Bag:
    """`bag` with features for every row."""
    rows, lookup = bag.distinct_tiles()
    if len(rows) == 0:
        raise ValueError("bag has no tiles to extract features from")

    distinct_feats = _run_model(extractor, _tile_locations(bag, rows), settings)

    feats = np.zeros((len(bag), distinct_feats.shape[1]), dtype=distinct_feats.dtype)
    feats[bag.mask] = distinct_feats[lookup]
    return dataclasses.replace(bag, feats=feats, extractor=str(extractor.identifier))


def _tile_locations(
    bag: Bag, rows: npt.NDArray[np.int64]
) -> list[tuple[str, TilePosition]]:
    """(cache file, position) of each row's tile image."""
    locations = []
    for row in rows:
        slide = bag.slides[bag.slide_idx[row]]
        if slide.cache_file is None:
            raise ValueError(
                f"slide {slide.slide_id} has no tile cache; run the tile step for it"
            )
        x, y = bag.coords_slide[row]
        locations.append((slide.cache_file, (float(x), float(y))))
    return locations


def _run_model(
    extractor: Extractor,
    locations: list[tuple[str, TilePosition]],
    settings: ExtractionSettings,
) -> npt.NDArray[np.float16]:
    """Features for each location, in order, as float16 (as STAMP stores them)."""
    dataset = _TileImages(locations, extractor.transform)
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        shuffle=False,
        pin_memory=settings.device.startswith("cuda"),
    )
    batches = []
    try:
        with torch.inference_mode():
            for tiles in loader:
                batches.append(extractor.model(tiles.to(settings.device)).half().cpu())
    finally:
        dataset.close()
    return torch.cat(batches).numpy()


class _TileImages(Dataset):
    """Transformed tile images, read from the tile cache."""

    def __init__(self, locations: list[tuple[str, TilePosition]], transform) -> None:
        self._locations = locations
        self._transform = transform
        self._reader = TileReader()

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, index: int) -> torch.Tensor:
        cache_file, position = self._locations[index]
        return self._transform(self._reader.read(cache_file, position))

    def close(self) -> None:
        self._reader.close()
