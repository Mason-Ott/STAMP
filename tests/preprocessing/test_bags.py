import numpy as np
import pytest

from stamp.preprocessing.data.bags import (
    Bag,
    PatientInfo,
    SamplingSummary,
    read_bag,
    write_bag,
)

from .builders import slide_info


def small_bag(n: int = 3, with_feats: bool = True) -> Bag:
    coords = np.stack([np.arange(n) * 10.0, np.zeros(n)], axis=1)
    feats = np.arange(n * 4, dtype=np.float16).reshape(n, 4) if with_feats else None
    return Bag.for_slide(
        slide_info(n_tiles=n),
        coords,
        tile_size_um=10.0,
        tile_size_px=8,
        feats=feats,
        extractor="fake" if with_feats else None,
    )


def test_write_then_read_gives_the_same_bag(tmp_path):
    bag = small_bag()
    bag = Bag(
        **{
            **bag.__dict__,
            "patient": PatientInfo("H_P", "P", "H", 1, missing_slide_ids=("X",)),
            "sampling": SamplingSummary({"n_tiles": 2}, 3, 2, [2]),
        }
    )
    write_bag(tmp_path / "bag.h5", bag)
    loaded = read_bag(tmp_path / "bag.h5")

    for field in ("coords", "coords_slide", "slide_idx", "tile_idx", "mask", "feats"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(bag, field))
    assert loaded.slides == bag.slides
    assert loaded.patient == bag.patient
    assert loaded.sampling == bag.sampling
    assert loaded.extractor == "fake"


def test_read_without_feats(tmp_path):
    write_bag(tmp_path / "bag.h5", small_bag())
    assert read_bag(tmp_path / "bag.h5", with_feats=False).feats is None


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("stamp.preprocessing.data.bags.json.dumps", broken)
    with pytest.raises(OSError):
        write_bag(tmp_path / "bag.h5", small_bag())
    assert list(tmp_path.iterdir()) == []


def test_zero_padding_rows_are_masked():
    bag = small_bag(2).with_zero_padding(3)
    assert len(bag) == 5
    assert bag.mask.tolist() == [True, True, False, False, False]
    assert (bag.feats[2:] == 0).all() and (bag.slide_idx[2:] == -1).all()


def test_distinct_tiles_collapses_repeats_in_bag_order():
    bag = small_bag(3).select(np.array([2, 0, 2, 1, 0]))
    rows, lookup = bag.distinct_tiles()
    assert rows.tolist() == [0, 1, 3]  # first occurrences of tiles 2, 0, 1
    assert lookup.tolist() == [0, 1, 0, 2, 1]
    np.testing.assert_array_equal(bag.feats[rows][lookup], bag.feats)


def test_distinct_tiles_skips_padding():
    rows, lookup = small_bag(2).with_zero_padding(2).distinct_tiles()
    assert rows.tolist() == [0, 1] and lookup.tolist() == [0, 1]


def test_inconsistent_shapes_are_rejected():
    bag = small_bag()
    with pytest.raises(ValueError, match="mask"):
        Bag(**{**bag.__dict__, "mask": np.ones(5, bool)})
    with pytest.raises(ValueError, match="slide_idx"):
        Bag(**{**bag.__dict__, "slide_idx": np.array([0, 1, 0], np.int32)})
