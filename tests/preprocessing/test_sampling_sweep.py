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
