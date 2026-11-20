"""Shared fixtures: synthetic slides, a clinical table, configs and fake models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.pipeline.config import PipelineConfig
from stamp.preprocessing.pipeline.plan import ModelLoaders
from stamp.preprocessing.steps.models import Lazy

from .fakes import FakeEncoder, fake_extractor

MPP = 0.5
TILE_SIZE_UM = 64.0
TILE_SIZE_PX = 32

# slide_id -> (height px, width px, tissue ellipses as (cy, cx, ry, rx))
SLIDES = {
    "S1": (1536, 2048, [(768, 1024, 700, 950)]),
    "S2": (1536, 2048, [(500, 600, 400, 500), (1100, 1500, 350, 450)]),
    "S3": (1024, 1024, [(512, 512, 200, 200)]),  # few tiles
    "S4": (1536, 2048, [(768, 1024, 700, 950)]),
    "S5": (1024, 1024, [(500, 500, 150, 150)]),  # few tiles
}

CLINICAL = pd.DataFrame(
    {
        # Patient P1 exists at two hospitals: they must stay two patients.
        "Patient": ["P1", "P1", "P1", "P2", "P2", "P9"],
        "StudyID/SlideID": ["S1", "S2", "S3", "S4", "S5", "S_no_label"],
        "isLNM": ["Y", "Y", "N", "n", "N", "?"],
        "hospital": ["H1", "H1", "H2", "H1", "H1", "H1"],
    }
)
LABELS = {"H1_P1": 1, "H1_P2": 0, "H2_P1": 0}


@dataclass(frozen=True)
class Dataset:
    wsi_dir: Path
    clinical_table: Path
    cache_dir: Path
    """Shared tile cache, so each test session tiles every slide only once."""


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    root = tmp_path_factory.mktemp("data")
    wsi_dir = root / "wsis"
    wsi_dir.mkdir()
    for seed, (slide_id, (height, width, ellipses)) in enumerate(SLIDES.items()):
        _write_slide(wsi_dir / f"{slide_id}.tiff", height, width, ellipses, seed)
    CLINICAL.to_csv(root / "clinical.csv", index=False)
    return Dataset(wsi_dir, root / "clinical.csv", root / "tile_cache")


@pytest.fixture
def make_configs(dataset: Dataset, tmp_path: Path):
    """Factory: (preprocessing config, pipeline config) writing under tmp_path."""

    def make(
        output_name: str = "out", **pipeline: object
    ) -> tuple[PreprocessingConfig, PipelineConfig]:
        extractor = pipeline.pop("extractor", "ctranspath")
        pre = PreprocessingConfig(
            output_dir=tmp_path / output_name,
            wsi_dir=dataset.wsi_dir,
            cache_dir=dataset.cache_dir,
            extractor=extractor,  # only names folders; the model is a fake
            tile_size_um=TILE_SIZE_UM,
            tile_size_px=TILE_SIZE_PX,
            device="cpu",
            max_workers=2,
        )
        settings = {
            "clinical_table": dataset.clinical_table,
            "num_workers": 0,
            "coordination": {"poll_interval_s": 0.2},
            **pipeline,
        }
        return pre, PipelineConfig.model_validate(settings)

    return make


@pytest.fixture
def fake_models() -> ModelLoaders:
    encoder = FakeEncoder()
    return ModelLoaders(extractor=Lazy(fake_extractor), encoder=Lazy(lambda: encoder))


def _write_slide(
    path: Path, height: int, width: int, ellipses: list, seed: int
) -> None:
    """A tiled TIFF with noisy pink 'tissue' ellipses on a near-white background."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 250, np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    for cy, cx, ry, rx in ellipses:
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        noise = rng.integers(0, 120, size=(height, width, 3), dtype=np.uint8)
        tissue = np.stack(
            [
                200 - noise[..., 0] // 2,
                80 + noise[..., 1] // 3,
                160 - noise[..., 2] // 3,
            ],
            axis=-1,
        )
        image[inside] = tissue[inside].astype(np.uint8)
    pixels_per_cm = 1e4 / MPP
    tifffile.imwrite(
        path,
        image,
        tile=(256, 256),
        photometric="rgb",
        compression="zlib",
        resolution=(pixels_per_cm, pixels_per_cm),
        resolutionunit="CENTIMETER",
    )
