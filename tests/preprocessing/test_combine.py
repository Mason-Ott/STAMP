import numpy as np
import pytest

from stamp.preprocessing.data.bags import Bag
from stamp.preprocessing.steps.combine import combine_slide_bags

from .builders import PATIENT, slide_info


def slide_bag(slide_id: str, coords: list[tuple[float, float]], **kwargs) -> Bag:
    return Bag.for_slide(
        slide_info(slide_id, len(coords)),
        np.array(coords, dtype=np.float64).reshape(-1, 2),
        tile_size_um=kwargs.pop("tile_size_um", 10.0),
        tile_size_px=8,
        **kwargs,
    )


def test_slides_are_laid_side_by_side_without_overlap():
    a = slide_bag("A", [(100, 50), (110, 50), (100, 60)])
    b = slide_bag("B", [(100, 50), (120, 70)])  # same positions as A on its own slide
    combined = combine_slide_bags([a, b], PATIENT, gap_tiles=1)

    assert combined.slide_idx.tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_array_equal(combined.coords_slide[:3], a.coords_slide)
    # A starts at the origin; B starts after A's width (20um) plus a one-tile gap.
    np.testing.assert_array_equal(
        combined.coords, [[0, 0], [10, 0], [0, 10], [30, 0], [50, 20]]
    )
    assert len(np.unique(combined.coords, axis=0)) == len(combined)
    assert [s.x_offset_um for s in combined.slides] == [-100, -70]


def test_offsets_map_virtual_back_to_slide_coordinates():
    combined = combine_slide_bags(
        [slide_bag("A", [(5, 5), (15, 5)]), slide_bag("B", [(40, 40)])],
        PATIENT,
        gap_tiles=2,
    )
    for k, slide in enumerate(combined.slides):
        rows = combined.slide_idx == k
        offset = np.array([slide.x_offset_um, slide.y_offset_um])
        np.testing.assert_array_equal(
            combined.coords[rows] - offset, combined.coords_slide[rows]
        )


def test_patient_and_tile_provenance_are_kept():
    combined = combine_slide_bags(
        [slide_bag("A", [(0, 0), (10, 0)]), slide_bag("B", [(0, 0)])],
        PATIENT,
        gap_tiles=1,
    )
    assert combined.patient == PATIENT
    assert [s.slide_id for s in combined.slides] == ["A", "B"]
    assert combined.tile_idx.tolist() == [0, 1, 0]


def test_empty_slide_keeps_its_index_but_takes_no_space():
    combined = combine_slide_bags(
        [slide_bag("A", []), slide_bag("B", [(30, 30)])], PATIENT, gap_tiles=1
    )
    assert combined.slide_idx.tolist() == [1]
    assert combined.coords.tolist() == [[0, 0]]
    assert [s.slide_id for s in combined.slides] == ["A", "B"]


def test_features_are_concatenated():
    a = slide_bag("A", [(0, 0)], feats=np.ones((1, 4), np.float16), extractor="x")
    b = slide_bag("B", [(0, 0)], feats=np.zeros((1, 4), np.float16), extractor="x")
    combined = combine_slide_bags([a, b], PATIENT, gap_tiles=1)
    assert combined.feats.tolist() == [[1] * 4, [0] * 4]
    assert combined.extractor == "x"


@pytest.mark.parametrize(
    "other, message",
    [
        (slide_bag("B", [(0, 0)], tile_size_um=20.0), "tiles of"),
        (
            slide_bag("B", [(0, 0)], feats=np.ones((1, 4), np.float16), extractor="x"),
            "different features",
        ),
    ],
)
def test_incompatible_slides_are_rejected(other, message):
    with pytest.raises(ValueError, match=message):
        combine_slide_bags([slide_bag("A", [(0, 0)]), other], PATIENT, gap_tiles=1)


def test_no_tiles_at_all_is_an_error():
    with pytest.raises(ValueError, match="none of the slides"):
        combine_slide_bags([slide_bag("A", [])], PATIENT, gap_tiles=1)
