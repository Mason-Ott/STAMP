"""The tile cache key includes canny_cutoff, and old caches can be migrated."""

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from stamp.preprocessing.data.bags import read_bag
from stamp.preprocessing.pipeline.config import StepName
from stamp.preprocessing.pipeline.plan import layout_for
from stamp.preprocessing.pipeline.run import run_pipeline_
from stamp.preprocessing.slides.migrate_cache import KNOWN_OLD_CODE_HASHES, migrate
from stamp.preprocessing.slides.tile_cache import PARAMS_MEMBER, cache_file_for


def cache_file(pre, slide: Path) -> Path:
    return cache_file_for(
        slide,
        cache_dir=pre.cache_dir,
        cache_tiles_ext=pre.cache_tiles_ext,
        tile_size_um=pre.tile_size_um,
        tile_size_px=pre.tile_size_px,
        brightness_cutoff=pre.brightness_cutoff,
        canny_cutoff=pre.canny_cutoff,
    )


def make_old_style_cache(new_cache: Path) -> Path:
    """Rewrite a cache zip as the tiler before the fix would have written it."""
    with ZipFile(new_cache) as zf:
        params = json.loads(zf.read(PARAMS_MEMBER))
        members = {
            name: zf.read(name) for name in zf.namelist() if name != PARAMS_MEMBER
        }
    del params["canny_cutoff"]
    params["code_sha256"] = next(iter(KNOWN_OLD_CODE_HASHES))
    old_key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    old_cache = new_cache.with_name(f"{Path(params['slide_path']).stem}.{old_key}.zip")

    with ZipFile(old_cache, "w") as zf:
        zf.writestr(PARAMS_MEMBER, json.dumps(params))
        for name, data in members.items():
            zf.writestr(name, data)
    new_cache.unlink()
    return old_cache


def test_canny_cutoff_is_part_of_the_cache_key(make_configs, dataset):
    slide = next(dataset.wsi_dir.glob("*.tiff"))
    default = cache_file(make_configs("a")[0], slide)
    for canny in (0.1, None):
        pre, _ = make_configs("b")
        pre = pre.model_copy(update={"canny_cutoff": canny})
        assert cache_file(pre, slide) != default


def test_old_caches_are_migrated_and_reused(
    make_configs, fake_models, dataset, tmp_path
):
    reference_pre, reference_cfg = make_configs("reference")
    run_pipeline_(reference_pre, reference_cfg, [StepName.TILE], models=fake_models)

    # A private copy of the cache, converted to the old naming scheme.
    pre, cfg = make_configs("migrated")
    pre = pre.model_copy(update={"cache_dir": tmp_path / "old_cache"})
    shutil.copytree(dataset.cache_dir, pre.cache_dir)
    slides = sorted(dataset.wsi_dir.glob("*.tiff"))
    for slide in slides:
        make_old_style_cache(cache_file(pre, slide))
    assert not any(cache_file(pre, slide).exists() for slide in slides)

    dry_run = migrate(pre.cache_dir, 0.02, dry_run=True, any_code_hash=False)
    assert dry_run["renamed"] == len(slides)
    assert not any(cache_file(pre, slide).exists() for slide in slides)

    assert (
        migrate(pre.cache_dir, 0.02, dry_run=False, any_code_hash=False)["renamed"] == 5
    )
    assert (
        migrate(pre.cache_dir, 0.02, dry_run=False, any_code_hash=False)["renamed"] == 0
    )
    assert all(cache_file(pre, slide).exists() for slide in slides)

    modified = {p: p.stat().st_mtime for p in pre.cache_dir.glob("*.zip")}
    run_pipeline_(pre, cfg, [StepName.TILE], models=fake_models)
    assert {p: p.stat().st_mtime for p in pre.cache_dir.glob("*.zip")} == modified

    reference = layout_for(reference_pre, reference_cfg).stage(StepName.TILE).folder
    for path in layout_for(pre, cfg).stage(StepName.TILE).folder.glob("*.h5"):
        np.testing.assert_array_equal(
            read_bag(path).coords, read_bag(reference / path.name).coords
        )


def test_a_different_canny_cutoff_gets_its_own_cache(make_configs, fake_models):
    pre, cfg = make_configs("strict")
    pre = pre.model_copy(update={"canny_cutoff": 0.2})
    run_pipeline_(pre, cfg, [StepName.TILE], models=fake_models)

    default_pre, default_cfg = make_configs("default")
    run_pipeline_(default_pre, default_cfg, [StepName.TILE], models=fake_models)

    def total_tiles(p, c) -> int:
        folder = layout_for(p, c).stage(StepName.TILE).folder
        return sum(len(read_bag(path)) for path in folder.glob("*.h5"))

    assert total_tiles(pre, cfg) < total_tiles(default_pre, default_cfg)
