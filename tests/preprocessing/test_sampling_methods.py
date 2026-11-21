"""Sampling strategies that look at positions, images or features."""

import numpy as np
import pytest
from PIL import Image

from stamp.preprocessing.data.bags import Bag
from stamp.preprocessing.pipeline.config import (
    STRATEGY_INPUT,
    PipelineConfig,
    SamplingConfig,
    StepName,
)
from stamp.preprocessing.pipeline.plan import layout_for
from stamp.preprocessing.pipeline.run import run_pipeline_
from stamp.preprocessing.steps.combine import combine_slide_bags
from stamp.preprocessing.steps.sample import patient_rng, sample_bag
from stamp.preprocessing.steps.sampling_methods import (
    glcm_features,
    grey_entropy,
    proportional_quota,
    representatives,
    rgb_histogram,
)

from .builders import PATIENT, slide_info
from .test_pipeline import read_folder


def config(**kwargs) -> PipelineConfig:
    return PipelineConfig.model_validate({"clinical_table": "unused.csv", **kwargs})


@pytest.mark.parametrize(
    "kwargs, valid",
    [
        ({"sampling": {"strategy": "spatial"}}, True),
        ({"sampling": {"strategy": "yottixel"}}, True),
        ({"sampling": {"strategy": "entropy"}}, True),
        ({"sampling": {"strategy": "texture"}}, True),
        ({"sampling": {"strategy": "conch_kmeans"}}, False),
        (
            {
                "order": "extract_then_sample",
                "sampling": {"strategy": "conch_kmeans"},
            },
            True,
        ),
        (
            {
                "order": "extract_then_sample",
                "slide_feat_dir": "/features",
                "sampling": {"strategy": "conch_kmeans"},
            },
            True,
        ),
        (
            {
                "order": "extract_then_sample",
                "slide_feat_dir": "/features",
                "sampling": {"strategy": "yottixel"},
            },
            False,
        ),
        (
            {
                "order": "extract_then_sample",
                "slide_feat_dir": "/features",
                "sampling": {"strategy": "spatial"},
            },
            True,
        ),
        ({"sampling": {"enabled": False, "strategy": "conch_kmeans"}}, True),
    ],
)
def test_strategies_are_only_valid_where_their_input_exists(kwargs, valid):
    if valid:
        config(**kwargs)
    else:
        with pytest.raises(ValueError, match="sampling strategy"):
            config(**kwargs)


def test_every_strategy_declares_what_it_needs():
    from typing import get_args

    from stamp.preprocessing.pipeline.config import SamplingStrategy

    assert set(STRATEGY_INPUT) == set(get_args(SamplingStrategy))


def test_conch_kmeans_needs_the_conch1_5_extractor(make_configs, fake_models):
    pre, cfg = make_configs(
        "wrong_extractor",
        order="extract_then_sample",
        sampling={"strategy": "conch_kmeans", "n_tiles": 20},
    )  # extractor: ctranspath
    with pytest.raises(ValueError, match="conch1_5"):
        run_pipeline_(pre, cfg, models=fake_models)


def test_tags_name_the_strategy_and_its_settings():
    tag = lambda **kw: SamplingConfig(n_tiles=50, **kw).tag  # noqa: E731
    assert tag(strategy="spatial") == "spatial-50-cell8"
    assert tag(strategy="spatial", spatial_cell_tiles=4) == "spatial-50-cell4"
    assert tag(strategy="yottixel") == "yottixel-50"
    assert tag(strategy="yottixel", color_clusters=5) == "yottixel-50-colors5"
    assert tag(strategy="entropy", seed=2) == "entropy-50"
    assert tag(strategy="texture") == "texture-50"
    assert tag(strategy="conch_kmeans") == "conch_kmeans-50"


# --- pure pieces -------------------------------------------------------------


def test_proportional_quota_splits_in_proportion_and_never_overfills():
    rng = np.random.default_rng(0)
    for _ in range(500):
        sizes = rng.integers(1, 200, size=rng.integers(1, 12))
        n = int(rng.integers(1, sizes.sum() + 1))
        quota = proportional_quota(sizes, n)
        assert quota.sum() == n and (quota <= sizes).all() and (quota >= 0).all()
    assert proportional_quota(np.array([90, 10]), 20).tolist() == [18, 2]


def test_representatives_are_distinct_and_cover_the_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10]])
    points = np.concatenate([c + rng.normal(0, 0.1, (30, 2)) for c in centres])
    chosen = representatives(points, 4, np.random.default_rng(1))
    assert len(set(chosen.tolist())) == 4
    assert sorted(int(i) // 30 for i in chosen) == [0, 1, 2, 3], "one per cluster"

    duplicates = np.zeros((10, 2))  # fewer distinct points than clusters
    filled = representatives(duplicates, 5, np.random.default_rng(1))
    assert len(set(filled.tolist())) == 5


def test_grey_entropy_is_zero_for_a_flat_tile_and_high_for_noise():
    flat = Image.new("RGB", (32, 32), (200, 200, 200))
    noise = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    )
    assert grey_entropy(flat) == 0.0
    assert grey_entropy(noise) > 7.0


def test_glcm_separates_flat_smooth_and_stripy_tiles():
    flat = Image.new("RGB", (64, 64), (120, 120, 120))
    stripes = np.zeros((64, 64, 3), np.uint8)
    stripes[(np.arange(64) // 4) % 2 == 0] = 255  # 4 px bands survive the 2x reduce
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    contrast = lambda image: glcm_features(image)[0]  # noqa: E731

    assert contrast(flat) == 0.0
    assert contrast(Image.fromarray(np.stack([ramp] * 3, axis=2))) < 1.0
    assert contrast(Image.fromarray(stripes)) > 50.0
    assert glcm_features(flat)[4] == 1.0, "flat tile: correlation defined as 1"


def test_rgb_histogram_is_normalised_and_tells_colours_apart():
    red = rgb_histogram(Image.new("RGB", (32, 32), (250, 10, 10)))
    blue = rgb_histogram(Image.new("RGB", (32, 32), (10, 10, 250)))
    assert red.sum() == pytest.approx(1.0) and red.shape == (24,)
    assert not np.allclose(red, blue)


# --- on bags -----------------------------------------------------------------


def slide_bag(coords, feats=None) -> Bag:
    n = len(coords)
    bag = Bag.for_slide(
        slide_info("S0", n),
        np.asarray(coords, dtype=float),
        tile_size_um=8.0,
        tile_size_px=8,
        feats=feats,
        extractor="fake" if feats is not None else None,
    )
    return combine_slide_bags([bag], PATIENT, gap_tiles=1)


def sample(bag: Bag, **settings) -> Bag:
    config = SamplingConfig(**settings)
    return sample_bag(bag, config, patient_rng(config.seed, "patient"))


def test_spatial_sampling_follows_the_density_of_tissue():
    dense = [(x * 8.0, y * 8.0) for x in range(10) for y in range(9)]  # 90 tiles
    sparse = [(1000.0 + x * 8.0, 1000.0) for x in range(10)]  # 10 tiles, far away
    bag = slide_bag(dense + sparse)
    sampled = sample(bag, strategy="spatial", n_tiles=20, spatial_cell_tiles=16)
    near = (sampled.coords_slide[:, 0] < 500).sum()
    assert len(sampled) == 20 and near == 18, "90% of the tiles, 90% of the sample"
    assert np.all(np.diff(sampled.tile_idx) > 0), "raster order kept"
    again = sample(bag, strategy="spatial", n_tiles=20, spatial_cell_tiles=16)
    np.testing.assert_array_equal(sampled.tile_idx, again.tile_idx)
    other = sample(bag, strategy="spatial", n_tiles=20, seed=1, spatial_cell_tiles=16)
    assert not np.array_equal(sampled.tile_idx, other.tile_idx)


def test_conch_kmeans_takes_one_tile_per_cluster_of_embeddings():
    rng = np.random.default_rng(0)
    directions = np.eye(3) * 5
    feats = np.concatenate(
        [d + rng.normal(0, 0.05, (20, 3)) for d in directions]
    ).astype(np.float16)
    coords = [(x * 8.0, 0.0) for x in range(60)]
    sampled = sample(slide_bag(coords, feats), strategy="conch_kmeans", n_tiles=3)
    assert sorted(int(i) // 20 for i in sampled.tile_idx) == [0, 1, 2]


def test_conch_kmeans_needs_features():
    bag = slide_bag([(x * 8.0, 0.0) for x in range(30)])
    with pytest.raises(ValueError, match="features"):
        sample(bag, strategy="conch_kmeans", n_tiles=5)


# --- end to end on the synthetic slides ---------------------------------------


@pytest.mark.parametrize("strategy", ["spatial", "entropy", "texture", "yottixel"])
def test_strategies_fill_each_slides_quota_with_distinct_tiles(
    make_configs, fake_models, strategy
):
    pre, cfg = make_configs(
        f"run_{strategy}", sampling={"strategy": strategy, "n_tiles": 20}
    )
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    reports = run_pipeline_(pre, cfg, steps, models=fake_models)
    assert all(r.ok for r in reports.values()), reports

    layout = layout_for(pre, cfg)
    full = read_folder(layout.stage(StepName.COMBINE).folder)
    sampled = read_folder(layout.stage(StepName.SAMPLE).folder)
    assert sampled.keys() == full.keys()
    checked = 0
    for key, bag in sampled.items():
        assert len(bag) <= 20
        assert sum(bag.sampling.per_slide) == len(bag)
        for k in range(len(bag.slides)):
            tiles = bag.tile_idx[bag.slide_idx == k]
            assert len(set(tiles.tolist())) == len(tiles), "no tile twice"
            assert np.all(np.diff(tiles) > 0), "raster order kept"
            assert len(tiles) == bag.sampling.per_slide[k]
            checked += len(tiles) < (full[key].slide_idx == k).sum()
    assert checked, "the test needs a slide with more tiles than its quota"


def test_entropy_keeps_the_highest_entropy_tiles(make_configs, fake_models):
    from stamp.preprocessing.slides.tile_cache import TileReader

    pre, cfg = make_configs(
        "entropy_check", sampling={"strategy": "entropy", "n_tiles": 20}
    )
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    run_pipeline_(pre, cfg, steps, models=fake_models)
    layout = layout_for(pre, cfg)
    full = read_folder(layout.stage(StepName.COMBINE).folder)
    reader = TileReader()
    compared = 0
    for key, bag in read_folder(layout.stage(StepName.SAMPLE).folder).items():
        for k, slide in enumerate(bag.slides):
            kept = {tuple(c) for c in bag.coords_slide[bag.slide_idx == k].tolist()}
            everything = full[key].coords_slide[full[key].slide_idx == k].tolist()
            score = {
                tuple(c): grey_entropy(reader.read(slide.cache_file, tuple(c)))
                for c in everything
            }
            dropped = [s for c, s in score.items() if c not in kept]
            if dropped:
                compared += 1
                assert min(score[c] for c in kept) >= max(dropped)
    reader.close()
    assert compared


def test_conch_kmeans_runs_after_extraction(make_configs, fake_models):
    pre, cfg = make_configs(
        "kmeans_run",
        order="extract_then_sample",
        extractor="conch1_5",
        sampling={"strategy": "conch_kmeans", "n_tiles": 20},
    )
    reports = run_pipeline_(pre, cfg, models=fake_models)
    assert all(r.ok for r in reports.values()), reports
    bags = read_folder(layout_for(pre, cfg).training_bags)
    assert bags and all(len(b) <= 20 and b.feats is not None for b in bags.values())
    assert max(len(b) for b in bags.values()) == 20
