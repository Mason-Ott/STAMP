"""End-to-end runs on synthetic slides, with a fake extractor and encoder."""

import sys
import types
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import pytest

from stamp.preprocessing.data.bags import Bag, read_bag
from stamp.preprocessing.pipeline.config import StepName
from stamp.preprocessing.pipeline.plan import active_steps, layout_for
from stamp.preprocessing.pipeline.run import run_pipeline_

from .conftest import LABELS
from .fakes import FEATURE_DIM, fake_extractor, tile_feature

N_TILES = 300  # larger than the synthetic patients, so padding is exercised


def read_folder(folder: Path) -> dict[str, Bag]:
    return {path.stem: read_bag(path) for path in sorted(folder.glob("*.h5"))}


def run(make_configs, fake_models, name="out", steps=None, **pipeline):
    pre, cfg = make_configs(name, **pipeline)
    reports = run_pipeline_(pre, cfg, steps, models=fake_models)
    assert not any(report.failed for report in reports.values()), reports
    return pre, cfg


@pytest.fixture
def sample_then_extract(make_configs, fake_models):
    pre, cfg = run(
        make_configs,
        fake_models,
        "ste",
        order="sample_then_extract",
        sampling={"n_tiles": N_TILES, "pad_mode": "zero"},
        encoder="titan",
        extractor="conch1_5",
        keep_intermediates=True,  # the tests compare against the full patient bags
    )
    layout = layout_for(pre, cfg)
    return {
        "layout": layout,
        "full": read_folder(layout.stage(StepName.COMBINE).folder),
        "bags": read_folder(layout.training_bags),
        "encoder": fake_models.encoder.get(),
    }


def test_one_bag_per_patient_with_label(sample_then_extract):
    bags = sample_then_extract["bags"]
    assert {key: bag.patient.label for key, bag in bags.items()} == LABELS
    for bag in bags.values():
        assert len(bag) == N_TILES and bag.feats.shape == (N_TILES, FEATURE_DIM)


def test_padding_rows_are_zero(sample_then_extract):
    for bag in sample_then_extract["bags"].values():
        assert (bag.feats[~bag.mask] == 0).all()
        assert (bag.slide_idx[~bag.mask] == -1).all()


def test_slides_do_not_overlap_in_virtual_space(sample_then_extract):
    for bag in sample_then_extract["bags"].values():
        real = bag.coords[bag.mask]
        assert len(np.unique(real, axis=0)) == len(real)
        right_edges = [
            real[bag.slide_idx[bag.mask] == k, 0].max() + bag.tile_size_um
            for k in range(len(bag.slides))
        ]
        left_edges = [
            real[bag.slide_idx[bag.mask] == k, 0].min() for k in range(len(bag.slides))
        ]
        assert all(
            r <= left for r, left in zip(right_edges, left_edges[1:], strict=False)
        )


def test_first_n_takes_the_first_tiles_of_each_slide(sample_then_extract):
    for key, bag in sample_then_extract["bags"].items():
        full = sample_then_extract["full"][key]
        for k, n in enumerate(bag.sampling.per_slide):
            chosen = bag.coords_slide[bag.mask & (bag.slide_idx == k)]
            np.testing.assert_array_equal(
                chosen, full.coords_slide[full.slide_idx == k][:n]
            )
            assert np.all(np.diff(chosen[:, 1]) >= 0), "not in raster order"


def test_no_tile_extends_past_the_slide_edge(sample_then_extract):
    for bag in sample_then_extract["bags"].values():
        for k, slide in enumerate(bag.slides):
            far = (bag.coords_slide[bag.slide_idx == k] + bag.tile_size_um) / slide.mpp
            assert (far[:, 0] <= slide.width_px + 1).all()
            assert (far[:, 1] <= slide.height_px + 1).all()


def test_features_come_from_the_right_place_on_the_right_slide(sample_then_extract):
    for bag in sample_then_extract["bags"].values():
        for row in np.flatnonzero(bag.mask)[::20]:
            slide = bag.slides[bag.slide_idx[row]]
            x, y = (bag.coords_slide[row] / slide.mpp).astype(int)
            size = round(bag.tile_size_um / slide.mpp)
            with openslide.open_slide(slide.slide_path) as wsi:
                region = wsi.read_region((x, y), 0, (size, size)).convert("RGB")
            expected = tile_feature(region.resize((bag.tile_size_px,) * 2))
            # The tile cache stores JPEGs, so allow for compression loss.
            np.testing.assert_allclose(bag.feats[row, :3], expected[:3], atol=0.02)


def test_encoder_sees_each_real_tile_once(sample_then_extract):
    calls = sample_then_extract["encoder"].calls
    bags = sample_then_extract["bags"]
    assert sorted(len(feats) for feats, _ in calls) == sorted(
        int(bag.mask.sum()) for bag in bags.values()
    )
    for path in sample_then_extract["layout"].embeddings.glob("*.h5"):
        with h5py.File(path) as f:
            assert f["feats"].shape == (FEATURE_DIM,)
            assert f.attrs["label"] == LABELS[path.stem]


def test_both_orders_give_identical_training_bags(
    make_configs, fake_models, sample_then_extract
):
    pre, cfg = run(
        make_configs,
        fake_models,
        "ets",
        order="extract_then_sample",
        sampling={"n_tiles": N_TILES, "pad_mode": "zero"},
    )
    other = read_folder(layout_for(pre, cfg).training_bags)
    assert other.keys() == sample_then_extract["bags"].keys()
    for key, bag in sample_then_extract["bags"].items():
        np.testing.assert_array_equal(bag.coords, other[key].coords)
        np.testing.assert_array_equal(bag.feats, other[key].feats)


def test_importing_plain_stamp_features_gives_identical_bags(
    make_configs, fake_models, sample_then_extract, monkeypatch, tmp_path
):
    from stamp.preprocessing import extract_

    # extract_ only accepts extractor names; point "ctranspath" at the fake.
    fake_module = types.SimpleNamespace(ctranspath=fake_extractor)
    monkeypatch.setitem(
        sys.modules, "stamp.preprocessing.extractor.ctranspath", fake_module
    )
    pre, _ = make_configs("plain")
    extract_(
        wsi_dir=pre.wsi_dir,
        output_dir=tmp_path / "plain_stamp",
        wsi_list=None,
        cache_dir=pre.cache_dir,
        cache_tiles_ext="jpg",
        extractor="ctranspath",
        tile_size_px=pre.tile_size_px,
        tile_size_um=pre.tile_size_um,
        max_workers=2,
        device="cpu",
        default_slide_mpp=None,
        brightness_cutoff=pre.brightness_cutoff,
        canny_cutoff=pre.canny_cutoff,
        generate_hash=False,
    )

    pre, cfg = run(
        make_configs,
        fake_models,
        "imported",
        order="extract_then_sample",
        slide_feat_dir=tmp_path / "plain_stamp" / "fake",
        sampling={"n_tiles": N_TILES, "pad_mode": "zero"},
    )
    imported = read_folder(layout_for(pre, cfg).training_bags)
    for key, bag in sample_then_extract["bags"].items():
        # plain STAMP stores coordinates as float32
        np.testing.assert_allclose(bag.coords, imported[key].coords, atol=1e-3)
        np.testing.assert_array_equal(bag.feats, imported[key].feats)


def test_repeat_padding_reuses_features(make_configs, fake_models, sample_then_extract):
    pre, cfg = run(
        make_configs,
        fake_models,
        "repeat",
        sampling={"n_tiles": N_TILES, "pad_mode": "repeat"},
    )
    for key, bag in read_folder(layout_for(pre, cfg).training_bags).items():
        zero_padded = sample_then_extract["bags"][key]
        n_real = int(zero_padded.mask.sum())
        assert len(bag) == N_TILES and bag.mask.all()
        np.testing.assert_array_equal(bag.feats[:n_real], zero_padded.feats[:n_real])
        np.testing.assert_array_equal(
            bag.feats[n_real : 2 * n_real], bag.feats[:n_real]
        )


def test_without_sampling_every_tile_is_extracted(
    make_configs, fake_models, sample_then_extract
):
    pre, cfg = run(make_configs, fake_models, "unsampled", sampling={"enabled": False})
    for key, bag in read_folder(layout_for(pre, cfg).training_bags).items():
        assert len(bag) == len(sample_then_extract["full"][key]) and bag.mask.all()


@pytest.mark.parametrize("allow_missing_slides", [False, True])
def test_patient_with_a_missing_slide(
    make_configs, fake_models, dataset, tmp_path, allow_missing_slides
):
    clinical = tmp_path / "clinical_missing.csv"
    pd.DataFrame(
        {
            "patient_id": ["P", "P"],
            "slide_id": ["S3", "not_scanned"],
            "label": ["1", "1"],
            "hospital": ["H", "H"],
        }
    ).to_csv(clinical, index=False)
    pre, cfg = make_configs(
        "missing",
        clinical_table=clinical,
        allow_missing_slides=allow_missing_slides,
        sampling={"enabled": False},
    )
    reports = run_pipeline_(
        pre, cfg, [StepName.TILE, StepName.COMBINE], models=fake_models
    )
    combined = layout_for(pre, cfg).stage(StepName.COMBINE).folder / "H_P.h5"
    if allow_missing_slides:
        assert reports[StepName.COMBINE].done == 1
        assert read_bag(combined).patient.missing_slide_ids == ("not_scanned",)
    else:
        assert reports[StepName.COMBINE].missing_inputs == 1
        assert not combined.exists()


def test_sampling_overviews_show_the_sampled_tiles(make_configs, fake_models):
    from PIL import Image

    pre, cfg = make_configs("overviews", sampling={"n_tiles": 50})
    pre = pre.model_copy(update={"save_visualisations": True})
    reports = run_pipeline_(
        pre, cfg, [StepName.TILE, StepName.COMBINE, StepName.SAMPLE], models=fake_models
    )
    assert all(report.ok for report in reports.values())

    layout = layout_for(pre, cfg)
    assert sorted(p.name for p in layout.tile_overviews.glob("*.png")) == [
        f"S{i}.png" for i in range(1, 6)
    ], "per-slide tile overviews"

    overviews = layout.sampling_overviews
    assert sorted(p.name for p in overviews.glob("*.png")) == [
        "H1_P1.png",
        "H1_P2.png",
        "H2_P1.png",
    ], "one image per patient"
    for patient in ("H1_P1", "H1_P2", "H2_P1"):
        image = np.asarray(Image.open(overviews / f"{patient}.png"))
        assert _colour_pixels(image, "green").any(), f"no sampled tiles: {patient}"
        assert _colour_pixels(image, "red").any(), f"no unsampled tiles: {patient}"


def _colour_pixels(image, colour):
    r, g, b = (image[..., i].astype(int) for i in range(3))
    if colour == "green":
        return (g > 120) & (r < 120) & (b < 120)
    return (r > 150) & (g < 120) & (b < 120)


def test_patient_overview_puts_slides_side_by_side_at_one_scale():
    from PIL import Image

    from stamp.preprocessing.data.bags import SlideInfo
    from stamp.preprocessing.slides.overview import SlidePanel, draw_patient_overview

    def panel(slide_id, width_px, height_px, kept, sampled):
        info = SlideInfo(
            slide_id=slide_id,
            slide_path="unused",
            mpp=1.0,
            width_px=width_px,
            height_px=height_px,
            cache_file=None,
            n_tiles=len(kept),
        )
        return SlidePanel(
            thumbnail=Image.new("RGB", (width_px, height_px), (240, 240, 240)),
            slide=info,
            kept_um=np.array(kept, dtype=float),
            sampled_um=np.array(sampled, dtype=float).reshape(-1, 2),
            label=slide_id,
        )

    small = panel("A", 600, 400, [[100, 100], [200, 100]], [[100, 100]])
    large = panel("B", 1200, 800, [[100, 100], [200, 100]], [])
    image = np.asarray(
        draw_patient_overview(
            [small, large], tile_size_um=100.0, title="P", max_dimension=500
        )
    )
    assert max(image.shape[:2]) <= 500 + 100  # title and label bars on top
    green = _colour_pixels(image, "green")
    red = _colour_pixels(image, "red")
    assert green.any() and red.any()
    # slide A is on the left and holds the only sampled tile
    assert (
        np.flatnonzero(green.any(axis=0)).max() < np.flatnonzero(red.any(axis=0)).max()
    )


def test_failed_overview_is_logged_and_does_not_fail_sampling(
    make_configs, fake_models, caplog
):
    pre, cfg = make_configs("broken_overview", sampling={"n_tiles": 50})
    pre = pre.model_copy(update={"save_visualisations": True})
    run_pipeline_(pre, cfg, [StepName.TILE, StepName.COMBINE], models=fake_models)
    # the bags are written; make the slides unreadable for the sample step
    from stamp.preprocessing.slides import overview

    def broken(*args, **kwargs):
        raise OSError("cannot open slide")

    overview_read = overview.read_thumbnail
    overview.read_thumbnail = broken
    try:
        with caplog.at_level("ERROR", logger="stamp"):
            reports = run_pipeline_(pre, cfg, [StepName.SAMPLE], models=fake_models)
    finally:
        overview.read_thumbnail = overview_read
    assert reports[StepName.SAMPLE].ok
    assert "failed to save sampling overview" in caplog.text


def folder_contents(root: Path) -> dict[str, int]:
    """Stage folder (relative to root) -> number of .h5 files in it."""
    return {
        str(folder.relative_to(root)): len(list(folder.glob("*.h5")))
        for folder in sorted(root.rglob("*"))
        if folder.is_dir() and any(folder.glob("*.h5"))
    }


def folder_names(pre, cfg) -> list[str]:
    layout = layout_for(pre, cfg)
    return [str(s.folder.relative_to(layout.root)) for s in layout.stages]


def test_stage_folders_are_named_by_the_settings_behind_them(make_configs):
    pre, cfg = make_configs(
        "ste", encoder="titan", extractor="conch1_5", sampling={"n_tiles": 50}
    )
    names = folder_names(pre, cfg)
    assert names[0].startswith("tile_positions/") and names[0].count("um_") == 1
    assert names[1].startswith("patient_positions/") and names[1].endswith("_gap1")
    assert names[2].startswith("sampled_positions/") and names[2].endswith(
        "/first_n-50"
    )
    assert names[3].startswith("features/conch1_5/") and names[3].endswith(
        "/first_n-50"
    )
    assert names[4].startswith("embeddings/titan/conch1_5/")

    pre, cfg = make_configs(
        "ets",
        encoder="titan",
        extractor="conch1_5",
        order="extract_then_sample",
        sampling={"n_tiles": 50},
    )
    other = folder_names(pre, cfg)
    assert other[1].startswith("slide_features/conch1_5/")
    assert other[2].startswith("patient_features/conch1_5/")
    assert other[3].startswith("features/conch1_5/") and other[3] == names[3]
    assert other[4] == names[4], "both orders end in the same folders"


def test_settings_that_shape_the_results_change_the_folders(make_configs):
    base_pre, base_cfg = make_configs("x", sampling={"n_tiles": 50})
    base = folder_names(base_pre, base_cfg)

    pre, cfg = make_configs("x", sampling={"n_tiles": 60})
    other = folder_names(pre, cfg)
    assert other[:2] == base[:2], "tiles and patients are shared"
    assert other[2:] != base[2:]

    pre, cfg = make_configs("x", sampling={"n_tiles": 50}, virtual_gap_tiles=2)
    other = folder_names(pre, cfg)
    assert other[0] == base[0] and other[1:] != base[1:]

    pre, cfg = make_configs("x", sampling={"n_tiles": 50}, drop_edge_tiles=False)
    assert not set(folder_names(pre, cfg)) & set(base)

    pre = base_pre.model_copy(update={"extractor": "conch1_5"})
    other = folder_names(pre, base_cfg)
    assert other[:3] == base[:3], "positions don't depend on the extractor"
    assert other[3] != base[3]


def test_only_the_results_remain_without_keep_intermediates(make_configs, fake_models):
    pre, cfg = run(
        make_configs,
        fake_models,
        "tidy",
        encoder="titan",
        extractor="conch1_5",
        keep_intermediates=False,
    )
    layout = layout_for(pre, cfg)
    assert folder_contents(pre.output_dir) == {
        str(layout.training_bags.relative_to(pre.output_dir)): 3,
        str(layout.embeddings.relative_to(pre.output_dir)): 3,
    }
    # Nothing is redone, although the intermediate files are gone.
    reports = run_pipeline_(pre, cfg, models=fake_models)
    assert all(r.done == 0 and r.ok for r in reports.values()), reports


def test_intermediates_are_kept_by_default(make_configs, fake_models):
    pre, cfg = run(make_configs, fake_models, "kept")
    assert sorted(folder_contents(pre.output_dir).values()) == [3, 3, 3, 5]
    assert {n.split("/")[0] for n in folder_contents(pre.output_dir)} == {
        "tile_positions",
        "patient_positions",
        "sampled_positions",
        "features",
    }


def test_runs_with_other_settings_share_one_output_dir(make_configs, fake_models):
    pre, cfg = run(make_configs, fake_models, "multi", sampling={"n_tiles": 50})
    first = layout_for(pre, cfg).training_bags
    first_bags = read_folder(first)

    # Other sampling settings: new sampled and feature folders, nothing re-tiled.
    other_cfg = cfg.model_copy(
        update={"sampling": cfg.sampling.model_copy(update={"n_tiles": 20})}
    )
    reports = run_pipeline_(pre, other_cfg, models=fake_models)
    assert not any(r.failed for r in reports.values())
    assert reports[StepName.TILE].done == 0 and reports[StepName.COMBINE].done == 0
    assert reports[StepName.SAMPLE].done == 3 and reports[StepName.EXTRACT].done == 3
    second = layout_for(pre, other_cfg).training_bags
    assert second != first
    assert all(len(b) <= 20 for b in read_folder(second).values())
    assert read_folder(first).keys() == first_bags.keys(), "first run untouched"

    # The other order ends in the same folder, which is already done.
    swapped = other_cfg.model_copy(update={"order": "extract_then_sample"})
    assert layout_for(pre, swapped).training_bags == second
    reports = run_pipeline_(pre, swapped, models=fake_models)
    assert reports[StepName.SAMPLE].done == 0 and reports[StepName.SAMPLE].ok

    # Another extractor: same positions, new features.
    other_pre = pre.model_copy(update={"extractor": "conch1_5"})
    reports = run_pipeline_(other_pre, cfg, models=fake_models)
    assert reports[StepName.SAMPLE].done == 0 and reports[StepName.EXTRACT].done == 3
    assert layout_for(other_pre, cfg).training_bags != first


def test_the_configuration_is_recorded_next_to_the_results(make_configs, fake_models):
    import yaml

    pre, cfg = run(
        make_configs, fake_models, "recorded", encoder="titan", extractor="conch1_5"
    )
    layout = layout_for(pre, cfg)
    sampled = layout.stage(StepName.SAMPLE).folder
    for folder in (sampled, layout.training_bags, layout.embeddings):
        recorded = yaml.safe_load((folder / "config.yaml").read_text())
        assert recorded["pipeline"]["sampling"]["n_tiles"] == cfg.sampling.n_tiles
        assert recorded["pipeline"]["sampling"]["seed"] == 0
        assert recorded["pipeline"]["encoder"] == "titan"
        assert recorded["preprocessing"]["extractor"] == "conch1_5"
        assert "workers" not in recorded["pipeline"]
    # tiling and patient folders are shared by runs with any sampling
    for step in (StepName.TILE, StepName.COMBINE):
        assert not (layout.stage(step).folder / "config.yaml").exists()


def test_sampling_only_runs_record_their_configuration_too(make_configs, fake_models):
    pre, cfg = make_configs("sampling_only", sampling={"n_tiles": 20})
    pre = pre.model_copy(update={"save_visualisations": True})
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    run_pipeline_(pre, cfg, steps, models=fake_models)
    layout = layout_for(pre, cfg)
    assert (layout.stage(StepName.SAMPLE).folder / "config.yaml").exists()
    assert (layout.sampling_overviews / "config.yaml").exists()
    assert not layout.training_bags.exists()


def test_the_seed_is_recorded_but_not_in_the_folder_name(make_configs, fake_models):
    import yaml

    seeded = {"strategy": "random", "n_tiles": 20, "seed": 5}
    pre, cfg = make_configs("seeded", sampling=seeded)
    assert "seed" not in layout_for(pre, cfg).training_bags.name
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    run_pipeline_(pre, cfg, steps, models=fake_models)
    folder = layout_for(pre, cfg).stage(StepName.SAMPLE).folder
    assert (
        yaml.safe_load((folder / "config.yaml").read_text())["pipeline"]["sampling"][
            "seed"
        ]
        == 5
    )

    # The same folder with another seed would mix up two different samples.
    other = cfg.model_copy(
        update={"sampling": cfg.sampling.model_copy(update={"seed": 6})}
    )
    assert layout_for(pre, other).stage(StepName.SAMPLE).folder == folder
    with pytest.raises(ValueError, match="different sampling settings.*seed: 5 -> 6"):
        run_pipeline_(pre, other, steps, models=fake_models)
    # ...and rerunning with the same seed is fine.
    reports = run_pipeline_(pre, cfg, steps, models=fake_models)
    assert all(r.ok and r.done == 0 for r in reports.values())


def test_failed_stage_keeps_its_input_for_recovery(
    make_configs, fake_models, monkeypatch
):
    from stamp.preprocessing.steps import extract

    pre, cfg = make_configs("recover", keep_intermediates=False)

    def broken(*args, **kwargs):
        raise RuntimeError("GPU fell over")

    monkeypatch.setattr(extract, "extract_features", broken)
    reports = run_pipeline_(pre, cfg, models=fake_models)
    assert reports[StepName.EXTRACT].failed == 3
    # The latest finished stage is still there...
    layout = layout_for(pre, cfg)
    sampled = layout.stage(StepName.SAMPLE).folder
    assert len(list(sampled.glob("*.h5"))) == 3
    assert not list(layout.stage(StepName.COMBINE).folder.glob("*.h5"))

    # ...so fixing the problem and retrying only redoes the failed stage.
    monkeypatch.undo()
    retry = cfg.model_copy(
        update={
            "coordination": cfg.coordination.model_copy(update={"retry_failed": True})
        }
    )
    reports = run_pipeline_(pre, retry, models=fake_models)
    assert reports[StepName.TILE].done == 0 and reports[StepName.EXTRACT].done == 3
    assert not list(sampled.glob("*.h5"))
    assert len(list(layout.training_bags.glob("*.h5"))) == 3


def test_encoding_is_skipped_when_the_encoder_cannot_use_the_extractor(
    make_configs, fake_models, caplog
):
    pre, cfg = make_configs("skip", encoder="titan")  # extractor: ctranspath
    assert StepName.ENCODE not in active_steps(pre, cfg)
    assert layout_for(pre, cfg).embeddings is None
    with caplog.at_level("WARNING", logger="stamp"):
        reports = run_pipeline_(pre, cfg, models=fake_models)
    assert StepName.ENCODE not in reports
    assert all(r.ok for r in reports.values())
    assert "skipping encoding" in caplog.text
    assert not list(pre.output_dir.rglob("*.failed"))
    assert not (pre.output_dir / "embeddings").exists()

    with pytest.raises(ValueError, match="can't use features"):
        run_pipeline_(pre, cfg, [StepName.ENCODE], models=fake_models)

    # With the extractor TITAN needs, encoding is part of the run.
    compatible = pre.model_copy(update={"extractor": "conch1_5"})
    assert StepName.ENCODE in active_steps(compatible, cfg)


def test_tissue_sampling_keeps_the_tiles_with_the_least_background(
    make_configs, fake_models
):
    from stamp.preprocessing.slides.tile_cache import TileReader
    from stamp.preprocessing.steps.sample import tissue_fraction

    n_tiles = 20
    pre, cfg = make_configs(
        "tissue", sampling={"strategy": "tissue", "n_tiles": n_tiles}
    )
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    reports = run_pipeline_(pre, cfg, steps, models=fake_models)
    assert all(r.ok for r in reports.values()), reports

    layout = layout_for(pre, cfg)
    full = read_folder(layout.stage(StepName.COMBINE).folder)
    sampled = read_folder(layout.stage(StepName.SAMPLE).folder)
    reader = TileReader()
    checked = 0
    for key, bag in sampled.items():
        assert len(bag) <= n_tiles
        for k, slide in enumerate(bag.slides):
            kept = {
                tuple(c)
                for c in bag.coords_slide[bag.mask & (bag.slide_idx == k)].tolist()
            }
            all_tiles = full[key].coords_slide[full[key].slide_idx == k].tolist()
            score = {
                tuple(c): tissue_fraction(reader.read(slide.cache_file, tuple(c)), 220)
                for c in all_tiles
            }
            dropped = [s for c, s in score.items() if c not in kept]
            if not dropped:
                continue
            checked += 1
            assert min(score[c] for c in kept) >= max(dropped), (key, slide.slide_id)
    assert checked, "the test needs a slide with more tiles than its quota"
    reader.close()
