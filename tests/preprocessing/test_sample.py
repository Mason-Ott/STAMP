from typing import get_args

import numpy as np
import pytest

from stamp.preprocessing.data.bags import Bag
from stamp.preprocessing.pipeline.config import SamplingConfig, SamplingStrategy
from stamp.preprocessing.steps.combine import combine_slide_bags
from stamp.preprocessing.steps.sample import SAMPLERS, allocate, patient_rng, sample_bag

from .builders import PATIENT, slide_info


@pytest.mark.parametrize(
    "available, budget, redistribute, expected",
    [
        ([10000, 10000], 8192, True, [4096, 4096]),
        ([10000, 10000, 10000], 8192, True, [2731, 2731, 2730]),
        ([100, 10000], 8192, True, [100, 8092]),
        ([100, 10000], 8192, False, [100, 4096]),
        ([100, 200, 10000], 8192, True, [100, 200, 7892]),
        ([10, 20], 8192, True, [10, 20]),
        ([0, 10000], 8192, True, [0, 8192]),
        ([0, 0], 10, True, [0, 0]),
    ],
)
def test_allocate(available, budget, redistribute, expected):
    assert allocate(available, budget, redistribute=redistribute).tolist() == expected


def test_allocate_never_exceeds_tiles_or_budget():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        available = rng.integers(0, 5000, size=rng.integers(1, 8))
        budget = int(rng.integers(1, 20000))
        taken = allocate(available, budget, redistribute=True)
        assert (taken <= available).all()
        assert taken.sum() == min(budget, available.sum())


def test_every_configurable_strategy_has_a_sampler():
    assert set(SAMPLERS) == set(get_args(SamplingStrategy))


def patient_bag(tiles_per_slide: list[int]) -> Bag:
    slides = []
    for k, n in enumerate(tiles_per_slide):
        coords = np.stack([np.arange(n) % 10 * 8.0, np.arange(n) // 10 * 8.0], axis=1)
        slides.append(
            Bag.for_slide(
                slide_info(f"S{k}", n),
                coords,
                tile_size_um=8.0,
                tile_size_px=8,
                feats=np.random.default_rng(k).random((n, 3)).astype(np.float16),
                extractor="fake",
            )
        )
    return combine_slide_bags(slides, PATIENT, gap_tiles=1)


def sample(bag: Bag, **settings) -> Bag:
    config = SamplingConfig(**settings)
    return sample_bag(bag, config, patient_rng(config.seed, "patient"))


def test_first_n_takes_the_first_tiles_of_each_slide():
    bag = patient_bag([30, 30])
    sampled = sample(bag, n_tiles=20)
    assert sampled.sampling.per_slide == [10, 10]
    for k in range(2):
        np.testing.assert_array_equal(
            sampled.tile_idx[sampled.slide_idx == k], np.arange(10)
        )


def test_random_is_reproducible_per_patient():
    bag = patient_bag([50, 50])
    a = sample(bag, n_tiles=20, strategy="random", seed=1)
    b = sample(bag, n_tiles=20, strategy="random", seed=1)
    c = sample(bag, n_tiles=20, strategy="random", seed=2)
    np.testing.assert_array_equal(a.tile_idx, b.tile_idx)
    assert not np.array_equal(a.tile_idx, c.tile_idx)
    # still sorted within each slide, so raster order is kept
    for k in range(2):
        assert np.all(np.diff(a.tile_idx[a.slide_idx == k]) > 0)


def test_small_patient_without_padding_keeps_all_tiles():
    sampled = sample(patient_bag([3, 4]), n_tiles=20)
    assert len(sampled) == 7 and sampled.mask.all()
    assert sampled.sampling.n_available == 7 and sampled.sampling.n_unique == 7


def test_zero_padding():
    sampled = sample(patient_bag([3, 4]), n_tiles=10, pad_mode="zero")
    assert len(sampled) == 10
    assert sampled.mask.sum() == 7
    assert (sampled.feats[~sampled.mask] == 0).all()


def test_repeat_padding():
    sampled = sample(patient_bag([3, 4]), n_tiles=10, pad_mode="repeat")
    assert len(sampled) == 10 and sampled.mask.all()
    assert sampled.sampling.n_unique == 7
    np.testing.assert_array_equal(sampled.feats[7:], sampled.feats[:3])


def test_sampling_already_sampled_bag_is_stable():
    once = sample(patient_bag([30, 30]), n_tiles=20)
    twice = sample(once, n_tiles=20)
    np.testing.assert_array_equal(once.tile_idx, twice.tile_idx)


def test_tissue_fraction_counts_pixels_below_the_cutoff():
    from PIL import Image

    from stamp.preprocessing.steps.sample import tissue_fraction

    tile = Image.new("RGB", (64, 64), (255, 255, 255))
    tile.paste((120, 60, 140), (0, 0, 64, 16))  # a quarter of the tile is stained
    assert tissue_fraction(tile, 220) == pytest.approx(0.25)
    assert tissue_fraction(tile, 20) == 0.0
    assert tissue_fraction(Image.new("RGB", (64, 64), (0, 0, 0)), 220) == 1.0


def test_tissue_settings_are_part_of_the_folder_tag():
    assert SamplingConfig(strategy="tissue", n_tiles=50).tag == "tissue-50"
    assert (
        SamplingConfig(strategy="tissue", background_cutoff=200).tag
        == "tissue-8192-bg200"
    )
