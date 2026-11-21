"""End-to-end runs on synthetic slides, with a fake extractor and encoder."""

import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import pytest

from stamp.preprocessing.data.bags import Bag, read_bag
from stamp.preprocessing.pipeline.config import PipelineConfig, StepName
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
    """Each stage's folder, relative to the experiment folder."""
    layout = layout_for(pre, cfg)
    return [str(s.folder.relative_to(layout.experiment_dir)) for s in layout.stages]


def test_results_are_organised_by_experiment_and_run(make_configs):
    pre, cfg = make_configs(
        "ste",
        experiment="exp1",
        encoder="titan",
        extractor="conch1_5",
        sampling={"n_tiles": 50},
    )
    layout = layout_for(pre, cfg)
    assert layout.experiment_dir == pre.output_dir / "exp1"
    assert layout.run_dir == pre.output_dir / "exp1" / "conch1_5-first_n-50"
    names = folder_names(pre, cfg)
    run = "conch1_5-first_n-50"
    assert names[0].startswith("shared/tile_positions/")
    assert names[1].startswith("shared/patient_positions/")
    assert names[2:] == [
        f"{run}/sampled_positions",
        f"{run}/features",
        f"{run}/embeddings/titan",
    ]
    assert layout.logs == layout.run_dir / "logs"
    assert layout.sampling_overviews == layout.run_dir / "overviews"

    pre, cfg = make_configs(
        "ets",
        experiment="exp1",
        run="my_run",
        order="extract_then_sample",
        encoder="titan",
        extractor="conch1_5",
        sampling={"n_tiles": 50},
    )
    other = folder_names(pre, cfg)
    assert other[0].startswith("shared/tile_positions/")
    assert other[1].startswith("shared/slide_features/conch1_5/")
    assert other[2].startswith("shared/patient_features/conch1_5/")
    assert other[3:] == ["my_run/features", "my_run/embeddings/titan"]


def test_settings_decide_whether_work_is_shared_or_per_run(make_configs):
    base_pre, base_cfg = make_configs("x", sampling={"n_tiles": 50})
    base = folder_names(base_pre, base_cfg)
    shared, run = base[:2], base[2:]

    pre, cfg = make_configs("x", sampling={"n_tiles": 60})
    other = folder_names(pre, cfg)
    assert other[:2] == shared, "sampling doesn't change the shared work"
    assert other[2:] != run, "a new run folder"

    pre, cfg = make_configs("x", sampling={"n_tiles": 50}, virtual_gap_tiles=2)
    other = folder_names(pre, cfg)
    assert other[0] == shared[0] and other[1] != shared[1]

    pre, cfg = make_configs("x", sampling={"n_tiles": 50}, drop_edge_tiles=False)
    assert set(folder_names(pre, cfg)[:2]).isdisjoint(shared)

    pre = base_pre.model_copy(update={"extractor": "conch1_5"})
    other = folder_names(pre, base_cfg)
    assert other[:2] == shared, "positions don't depend on the extractor"
    assert other[2:] != run

    pre, cfg = make_configs("x", sampling={"n_tiles": 50}, experiment="other")
    assert (
        layout_for(pre, cfg).experiment_dir
        != layout_for(base_pre, base_cfg).experiment_dir
    )


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
    layout = layout_for(pre, cfg)
    folders = {
        Path(k).parts[-1]: v for k, v in folder_contents(layout.experiment_dir).items()
    }
    assert folders["sampled_positions"] == folders["features"] == 3
    assert sorted(folder_contents(layout.experiment_dir).values()) == [3, 3, 3, 5]
    assert len(list(layout.shared_dir.rglob("*.h5"))) == 8, "tiles and patients"


def test_runs_of_one_experiment_share_work_but_keep_their_results_apart(
    make_configs, fake_models
):
    pre, cfg = run(
        make_configs, fake_models, "multi", run="a", sampling={"n_tiles": 50}
    )
    first = layout_for(pre, cfg)
    first_bags = read_folder(first.training_bags)

    # Another run: new run folder, but nothing re-tiled or re-combined.
    other_cfg = cfg.model_copy(
        update={
            "run": "b",
            "sampling": cfg.sampling.model_copy(update={"n_tiles": 20}),
        }
    )
    reports = run_pipeline_(pre, other_cfg, models=fake_models)
    assert not any(r.failed for r in reports.values())
    assert reports[StepName.TILE].done == 0 and reports[StepName.COMBINE].done == 0
    assert reports[StepName.SAMPLE].done == 3 and reports[StepName.EXTRACT].done == 3
    second = layout_for(pre, other_cfg)
    assert second.run_dir != first.run_dir
    assert second.stage(StepName.TILE).folder == first.stage(StepName.TILE).folder
    assert all(len(b) <= 20 for b in read_folder(second.training_bags).values())
    assert read_folder(first.training_bags).keys() == first_bags.keys()
    assert all(len(b) <= 50 for b in read_folder(first.training_bags).values())

    # Another extractor in the same experiment: the shared positions are reused.
    other_pre = pre.model_copy(update={"extractor": "conch1_5"})
    third_cfg = cfg.model_copy(update={"run": "c"})
    reports = run_pipeline_(other_pre, third_cfg, models=fake_models)
    assert reports[StepName.TILE].done == 0 and reports[StepName.EXTRACT].done == 3


def test_each_run_has_its_config_and_a_guide(make_configs, fake_models):
    import yaml

    pre, cfg = run(
        make_configs,
        fake_models,
        "recorded",
        run="first_run",
        encoder="titan",
        extractor="conch1_5",
    )
    layout = layout_for(pre, cfg)
    recorded = yaml.safe_load((layout.run_dir / "config.yaml").read_text())
    assert recorded["pipeline"]["sampling"]["n_tiles"] == cfg.sampling.n_tiles
    assert recorded["pipeline"]["sampling"]["seed"] == 0
    assert recorded["pipeline"]["encoder"] == "titan"
    assert recorded["pipeline"]["run"] == "first_run"
    assert recorded["preprocessing"]["extractor"] == "conch1_5"
    assert "workers" not in recorded["pipeline"]

    guide = (layout.run_dir / "README.md").read_text()
    assert "Run `first_run` of experiment `test`" in guide
    positions = [guide.index(f". {s.step}:") for s in layout.stages]
    assert positions == sorted(positions), "steps in order"
    assert "../shared/tile_positions/" in guide and "embeddings/titan/" in guide
    assert "first_run/" in guide and "config.yaml" in guide

    # a second run joins the experiment's index
    run_pipeline_(
        pre,
        cfg.model_copy(update={"run": "second_run"}),
        [StepName.TILE, StepName.COMBINE],
        models=fake_models,
    )
    index = (layout.experiment_dir / "README.md").read_text()
    assert "first_run" in index and "second_run" in index and "first_n" in index


def test_sampling_only_runs_have_their_config_and_overviews(make_configs, fake_models):
    pre, cfg = make_configs("sampling_only", sampling={"n_tiles": 20})
    pre = pre.model_copy(update={"save_visualisations": True})
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    run_pipeline_(pre, cfg, steps, models=fake_models)
    layout = layout_for(pre, cfg)
    assert (layout.run_dir / "config.yaml").exists()
    assert (layout.run_dir / "README.md").exists()
    assert list(layout.sampling_overviews.glob("*.png"))
    assert not layout.training_bags.exists()


def test_a_run_folder_rejects_other_result_shaping_settings(make_configs, fake_models):
    seeded = {"strategy": "random", "n_tiles": 20, "seed": 5}
    pre, cfg = make_configs("seeded", run="r", sampling=seeded)
    steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE]
    run_pipeline_(pre, cfg, steps, models=fake_models)

    other = cfg.model_copy(
        update={"sampling": cfg.sampling.model_copy(update={"seed": 6})}
    )
    assert layout_for(pre, other).run_dir == layout_for(pre, cfg).run_dir
    with pytest.raises(ValueError, match="different settings.*sampling.seed: 5 -> 6"):
        run_pipeline_(pre, other, steps, models=fake_models)
    # nothing was changed by the rejected run
    import yaml

    recorded = yaml.safe_load(
        (layout_for(pre, cfg).run_dir / "config.yaml").read_text()
    )
    assert recorded["pipeline"]["sampling"]["seed"] == 5

    # the same settings again are fine, and so is adding an encoder later
    reports = run_pipeline_(pre, cfg, steps, models=fake_models)
    assert all(r.ok and r.done == 0 for r in reports.values())
    with_encoder = cfg.model_copy(update={"encoder": "titan"})
    run_pipeline_(pre, with_encoder, steps, models=fake_models)


def test_experiment_and_run_names_are_validated(make_configs):
    for bad in ("", "has space", "../escape", ".hidden", "shared"):
        with pytest.raises(ValueError, match="pipeline.(experiment|run)"):
            make_configs("x", experiment=bad)
        with pytest.raises(ValueError, match="pipeline.(experiment|run)"):
            make_configs("x", run=bad)
    with pytest.raises(ValueError, match="experiment"):
        PipelineConfig.model_validate({"clinical_table": "c.csv"})


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


def test_chunked_encoding_averages_chunk_embeddings_of_all_tiles(
    make_configs, fake_models
):
    pre, cfg = make_configs(
        "chunked",
        encoder="titan",
        extractor="conch1_5",
        sampling={"enabled": False},
        encoding={"chunk_size": 20, "seed": 3},
    )
    reports = run_pipeline_(pre, cfg, models=fake_models)
    assert all(r.ok for r in reports.values()), reports

    layout = layout_for(pre, cfg)
    assert layout.embeddings.name == "chunk20-seed3"
    bags = read_folder(layout.training_bags)
    calls = fake_models.encoder.get().calls
    assert all(len(feats) <= 20 for feats, _ in calls)

    # Every tile of every patient goes through the encoder exactly once.
    assert sum(len(feats) for feats, _ in calls) == sum(
        len(bag.distinct_tiles()[0]) for bag in bags.values()
    )
    big = max(bags.values(), key=len)
    assert len(big.distinct_tiles()[0]) > 20, "the test needs a patient with chunks"

    for key, bag in bags.items():
        with h5py.File(layout.embeddings / f"{key}.h5") as f:
            embedding, n_chunks = f["feats"][:], f.attrs["n_chunks"]
        rows, _ = bag.distinct_tiles()
        assert n_chunks == -(-len(rows) // 20)
        # fake embedding = mean tile feature; mean of equal-size chunk means
        # is the mean over all tiles when chunks have equal size, else close to it
        np.testing.assert_allclose(
            embedding, bag.feats[rows].mean(0), rtol=0.05, atol=0.05
        )


def test_chunks_are_random_but_reproducible_and_cover_the_bag(sample_then_extract):
    from stamp.preprocessing.steps.encode import EncodingSettings, chunks_of

    bag = next(b for b in sample_then_extract["bags"].values() if len(b) > 40)
    rows = bag.distinct_tiles()[0]
    settings = EncodingSettings(device="cpu", chunk_size=25, seed=1)
    chunks = chunks_of(bag, settings)
    again = chunks_of(bag, settings)
    other = chunks_of(bag, EncodingSettings(device="cpu", chunk_size=25, seed=2))
    assert all(len(c) <= 25 for c in chunks)
    assert sorted(np.concatenate(chunks)) == sorted(rows)
    assert all(np.array_equal(a, b) for a, b in zip(chunks, again, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(chunks, other, strict=True))
    whole = chunks_of(bag, EncodingSettings(device="cpu", chunk_size=10**6))
    assert len(whole) == 1 and np.array_equal(whole[0], rows)


def test_chunking_settings_name_the_embedding_folder(make_configs):
    base_pre, base_cfg = make_configs("x", encoder="titan", extractor="conch1_5")
    pre, cfg = make_configs(
        "x", encoder="titan", extractor="conch1_5", encoding={"chunk_size": 100}
    )
    base, chunked = folder_names(base_pre, base_cfg), folder_names(pre, cfg)
    assert chunked[:-1] == base[:-1], "only the embeddings differ"
    assert chunked[-1] == f"{base[-1]}/chunk100-seed0"

    with pytest.raises(ValueError, match="need pipeline.encoder"):
        make_configs("x", encoding={"chunk_size": 100})


SLIDE_COUNTS = {"H1_P1": 2, "H1_P2": 2, "H2_P1": 1}


def per_slide_run(make_configs, fake_models, name, **encoding):
    pre, cfg = make_configs(
        name,
        encoder="titan",
        extractor="conch1_5",
        sampling={"enabled": False},
        encoding=encoding,
    )
    reports = run_pipeline_(pre, cfg, models=fake_models)
    assert all(r.ok for r in reports.values()), reports
    return pre, cfg


def read_embeddings(folder: Path) -> dict[str, np.ndarray]:
    result = {}
    for path in sorted(folder.glob("*.h5")):
        with h5py.File(path) as f:
            result[path.stem] = f["feats"][:]
    return result


def test_per_slide_embeddings_are_aggregated(make_configs, fake_models):
    pre, cfg = per_slide_run(
        make_configs, fake_models, "slides", slide_aggregation="mean"
    )
    layout = layout_for(pre, cfg)
    assert layout.stage(StepName.ENCODE).folder.parts[-2] == "slide_embeddings"
    assert layout.embeddings.name == "slides_mean"

    calls = fake_models.encoder.get().calls
    assert len(calls) == sum(SLIDE_COUNTS.values()), "one encoder call per slide"
    bags = read_folder(layout.training_bags)

    with h5py.File(layout.stage(StepName.ENCODE).folder / "H1_P1.h5") as f:
        slide_feats = f["slide_feats"][:]
        assert json.loads(f.attrs["slide_ids"]) == ["S1", "S2"]
    assert slide_feats.shape == (2, FEATURE_DIM)
    # fake embedding = mean tile feature of the slide's own tiles
    bag = bags["H1_P1"]
    for k, row in enumerate(slide_feats):
        np.testing.assert_allclose(
            row, bag.feats[bag.slide_idx == k].mean(0), rtol=1e-2, atol=1e-3
        )

    embeddings = read_embeddings(layout.embeddings)
    assert set(embeddings) == set(SLIDE_COUNTS)
    np.testing.assert_allclose(embeddings["H1_P1"], slide_feats.mean(0), rtol=1e-5)
    with h5py.File(layout.embeddings / "H1_P1.h5") as f:
        assert f.attrs["feat_type"] == "patient" and f.attrs["n_slides"] == 2
        assert f.attrs["slide_aggregation"] == "mean"


def test_trying_another_aggregation_only_reruns_the_aggregate_step(
    make_configs, fake_models
):
    pre, cfg = per_slide_run(make_configs, fake_models, "agg", slide_aggregation="mean")
    n_calls = len(fake_models.encoder.get().calls)
    first = layout_for(pre, cfg)

    other = cfg.model_copy(
        update={
            "encoding": cfg.encoding.model_copy(update={"slide_aggregation": "max"})
        }
    )
    reports = run_pipeline_(pre, other, models=fake_models)
    assert reports[StepName.ENCODE].done == 0 and reports[StepName.AGGREGATE].done == 3
    assert len(fake_models.encoder.get().calls) == n_calls, "nothing re-encoded"

    second = layout_for(pre, other)
    assert second.stage(StepName.ENCODE).folder == first.stage(StepName.ENCODE).folder
    assert second.embeddings != first.embeddings
    with h5py.File(first.stage(StepName.ENCODE).folder / "H1_P1.h5") as f:
        slide_feats = f["slide_feats"][:]
    np.testing.assert_array_equal(
        read_embeddings(second.embeddings)["H1_P1"], slide_feats.max(0)
    )
    np.testing.assert_allclose(
        read_embeddings(first.embeddings)["H1_P1"], slide_feats.mean(0), rtol=1e-5
    )


def test_concatenated_slide_embeddings_are_padded_to_max_slides(
    make_configs, fake_models
):
    pre, cfg = per_slide_run(
        make_configs,
        fake_models,
        "concat",
        slide_aggregation="concat",
        max_slides=3,
    )
    layout = layout_for(pre, cfg)
    assert layout.embeddings.name == "slides_concat3"
    embeddings = read_embeddings(layout.embeddings)
    assert {e.shape for e in embeddings.values()} == {(3 * FEATURE_DIM,)}
    with h5py.File(layout.stage(StepName.ENCODE).folder / "H1_P1.h5") as f:
        slide_feats = f["slide_feats"][:]
    np.testing.assert_array_equal(
        embeddings["H1_P1"],
        np.concatenate([slide_feats.reshape(-1), np.zeros(FEATURE_DIM)]),
    )
    single = embeddings["H2_P1"].reshape(3, FEATURE_DIM)
    assert single[0].any() and not single[1:].any()


def test_too_few_slide_slots_is_rejected_before_anything_runs(
    make_configs, fake_models
):
    pre, cfg = make_configs(
        "slots",
        encoder="titan",
        extractor="conch1_5",
        encoding={"slide_aggregation": "concat", "max_slides": 1},
    )
    with pytest.raises(ValueError, match="more than encoding.max_slides"):
        run_pipeline_(pre, cfg, models=fake_models)
    assert not (pre.output_dir / "slide_embeddings").exists()


def test_slide_aggregation_settings_are_validated(make_configs):
    with pytest.raises(ValueError, match="max_slides is needed for"):
        make_configs("x", encoder="titan", encoding={"slide_aggregation": "concat"})
    with pytest.raises(ValueError, match="only for"):
        make_configs(
            "x",
            encoder="titan",
            encoding={"slide_aggregation": "mean", "max_slides": 2},
        )
    with pytest.raises(ValueError, match="need pipeline.encoder"):
        make_configs("x", encoding={"slide_aggregation": "mean"})


def test_per_slide_encoding_with_chunks_averages_chunks_within_each_slide(
    make_configs, fake_models
):
    pre, cfg = per_slide_run(
        make_configs,
        fake_models,
        "slide_chunks",
        slide_aggregation="mean",
        chunk_size=15,
    )
    layout = layout_for(pre, cfg)
    assert layout.embeddings.parts[-2:] == ("chunk15-seed0", "slides_mean")
    calls = fake_models.encoder.get().calls
    assert all(len(feats) <= 15 for feats, _ in calls)
    bags = read_folder(layout.training_bags)
    assert sum(len(f) for f, _ in calls) == sum(
        len(b.distinct_tiles()[0]) for b in bags.values()
    )
    assert len(calls) > sum(SLIDE_COUNTS.values()), "some slides needed chunks"
