"""scripts/sampling_sweep.py: one config per strategy, one host each."""

import importlib.util
from pathlib import Path

import pytest
import yaml

from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.pipeline.config import PipelineConfig

SCRIPT = Path(__file__).parents[2] / "scripts" / "sampling_sweep.py"

spec = importlib.util.spec_from_file_location("sampling_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

BASE = {
    "preprocessing": {
        "output_dir": "/out",
        "wsi_dir": "/wsis",
        "extractor": "conch1_5",
        "save_visualisations": False,
    },
    "pipeline": {
        "clinical_table": "/clinical.csv",
        "order": "extract_then_sample",
        "sampling": {"n_tiles": 8192, "strategy": "first_n", "pad_mode": "repeat"},
        "encoder": "titan",
        "workers": {
            "hosts": ["h1", "h2", "h3"],
            "project_dir": "/project",
            "activate": "source env",
        },
    },
}


def test_every_strategy_gets_its_own_valid_config_and_a_host_in_turn():
    strategies = list(sweep.STEPS)
    variants = sweep.plan_variants(BASE, strategies, 1024)
    assert [v.strategy for v in variants] == strategies
    assert [v.host for v in variants[:4]] == ["h1", "h2", "h3", "h1"], "wraps around"

    for variant in variants:
        # every generated config is a valid pipeline config
        pre = PreprocessingConfig.model_validate(variant.config["preprocessing"])
        cfg = PipelineConfig.model_validate(variant.config["pipeline"])
        assert pre.save_visualisations
        assert cfg.sampling.strategy == variant.strategy
        assert cfg.sampling.n_tiles == 1024
        assert cfg.sampling.pad_mode == "repeat", "other sampling settings kept"
        assert cfg.encoder is None
        assert cfg.workers.hosts == [variant.host]
        assert cfg.workers.project_dir == "/project"
        expected = (
            "extract_then_sample"
            if variant.strategy == "conch_kmeans"
            else ("sample_then_extract")
        )
        assert str(cfg.order) == expected


def test_the_base_config_is_not_changed():
    before = yaml.safe_dump(BASE)
    sweep.plan_variants(BASE, ["entropy"], 1024)
    assert yaml.safe_dump(BASE) == before


def test_hosts_can_be_overridden_and_are_required():
    variants = sweep.plan_variants(BASE, ["entropy", "random"], 1024, ["x", "y"])
    assert [v.host for v in variants] == ["x", "y"]
    without_workers = {"preprocessing": BASE["preprocessing"], "pipeline": {}}
    with pytest.raises(ValueError, match="workers.hosts"):
        sweep.plan_variants(without_workers, ["entropy"], 1024)


def test_configs_are_written_one_file_per_strategy(tmp_path):
    variants = sweep.plan_variants(BASE, ["entropy", "random"], 1024)
    paths = sweep.write_configs(variants, tmp_path / "sweep")
    assert sorted(p.name for p in paths.values()) == [
        "sampling_entropy.yaml",
        "sampling_random.yaml",
    ]
    written = yaml.safe_load(paths["entropy"].read_text())
    assert written["pipeline"]["sampling"]["strategy"] == "entropy"


def test_working_hosts_keeps_pool_order_and_drops_failures():
    problems = {"h1": None, "h2": "Permission denied", "h3": None, "h4": "timed out"}
    assert sweep.working_hosts(["h1", "h2", "h3", "h4", "h5"], problems) == [
        "h1",
        "h3",
        "h5",
    ]


def test_a_strategy_whose_host_fails_moves_to_the_next_spare_host():
    variants = sweep.plan_variants(BASE, ["entropy", "random"], 1024, ["h1", "h2"])
    started = []

    def starter(variant):
        started.append((variant.strategy, variant.host))
        return "failed: ssh: boom" if variant.host == "h1" else "started"

    results = sweep.start_all(variants, spares=["h3", "h4"], starter=starter)
    by_strategy = {v.strategy: (v.host, outcome) for v, outcome in results}
    assert by_strategy == {"entropy": ("h3", "started"), "random": ("h2", "started")}
    assert ("entropy", "h1") in started, "tried the first host before moving on"
    moved = next(v for v, _ in results if v.strategy == "entropy")
    assert moved.config["pipeline"]["workers"]["hosts"] == ["h3"]


def test_without_spares_a_failure_is_reported():
    variants = sweep.plan_variants(BASE, ["entropy"], 1024, ["h1"])
    results = sweep.start_all(variants, spares=[], starter=lambda v: "failed: down")
    assert [(v.host, o) for v, o in results] == [("h1", "failed: down")]


def test_several_failures_use_spares_in_order_and_stop_when_they_run_out():
    variants = sweep.plan_variants(
        BASE, ["entropy", "random", "tissue"], 1024, ["a", "b", "c"]
    )
    bad = {"a", "b", "c", "d"}
    results = sweep.start_all(
        variants,
        spares=["d", "e"],
        starter=lambda v: "failed: x" if v.host in bad else "started",
    )
    outcomes = {v.strategy: (v.host, o) for v, o in results}
    assert outcomes["entropy"] == ("e", "started")
    assert outcomes["random"][1].startswith("failed"), "no spares left"
    assert outcomes["tissue"][1].startswith("failed")


def test_status_and_stop_use_the_hosts_written_by_start(tmp_path):
    variants = sweep.plan_variants(BASE, ["entropy", "random"], 1024, ["h1", "h2"])
    moved = [sweep.with_host(variants[0], "h9"), variants[1]]
    sweep.write_configs(moved, tmp_path)
    loaded = sweep.load_variants(tmp_path, ["entropy", "random"])
    assert [(v.strategy, v.host) for v in loaded] == [
        ("entropy", "h9"),
        ("random", "h2"),
    ]
    with pytest.raises(SystemExit, match="run `start` first"):
        sweep.load_variants(tmp_path, ["tissue"])
