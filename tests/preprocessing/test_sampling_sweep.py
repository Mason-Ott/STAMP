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
        "experiment": "sweep",
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


def allocate(strategies, pool, outcomes):
    """Run `allocate` with a fake starter; `outcomes` maps host -> outcome."""
    variants = sweep.plan_variants(BASE, strategies, 1024, pool)
    tried, reported = [], []

    def starter(variant):
        tried.append((variant.strategy, variant.host))
        return outcomes.get(variant.host, "started")

    results = sweep.allocate(
        variants, pool, starter, lambda v, o: reported.append((v.host, o))
    )
    return [(v.strategy, v.host, o) for v, o in results], tried, reported


def test_each_strategy_takes_the_next_host_of_the_pool():
    results, tried, _ = allocate(["entropy", "random"], ["h1", "h2", "h3"], {})
    assert results == [("entropy", "h1", "started"), ("random", "h2", "started")]
    assert tried == [("entropy", "h1"), ("random", "h2")], "h3 stays a spare"


def test_a_failing_host_is_skipped_and_the_next_one_tried():
    outcomes = {"h1": "failed: Permission denied", "h2": "failed: No route to host"}
    results, tried, reported = allocate(
        ["entropy", "random"], ["h1", "h2", "h3", "h4"], outcomes
    )
    assert results == [("entropy", "h3", "started"), ("random", "h4", "started")]
    assert tried == [
        ("entropy", "h1"),
        ("entropy", "h2"),
        ("entropy", "h3"),
        ("random", "h4"),
    ]
    assert reported == [
        ("h1", "failed: Permission denied"),
        ("h2", "failed: No route to host"),
    ]


def test_a_host_is_never_given_two_strategies():
    results, _, _ = allocate(
        ["entropy", "random", "tissue"], ["h1", "h2"], {"h1": "already running"}
    )
    hosts = [
        host
        for _, host, outcome in results
        if outcome in ("started", "already running")
    ]
    assert len(hosts) == len(set(hosts)) == 2
    assert results[2] == ("tissue", "", "failed: no host left in the pool")


def test_a_pool_that_runs_out_leaves_the_rest_without_a_host():
    results, tried, _ = allocate(
        ["entropy", "random"], ["h1", "h2"], {"h1": "failed: x", "h2": "failed: y"}
    )
    assert results[0][1:] == ("h2", "failed: y")
    assert results[1] == ("random", "", "failed: no host left in the pool")
    assert [h for _, h in tried] == ["h1", "h2"]


def test_the_moved_strategy_has_its_new_host_in_the_config():
    variants = sweep.plan_variants(BASE, ["entropy"], 1024, ["h1"])
    seen = []
    results = sweep.allocate(
        variants,
        ["h1", "h2"],
        lambda v: (
            seen.append(v.config["pipeline"]["workers"]["hosts"])
            or ("failed" if v.host == "h1" else "started")
        ),
    )
    assert seen == [["h1"], ["h2"]]
    assert results[0][0].config["pipeline"]["workers"]["hosts"] == ["h2"]


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


def test_a_sweep_is_one_experiment_with_a_run_per_strategy(tmp_path):
    variants = sweep.plan_variants(BASE, ["entropy", "random"], 1024)
    for variant in variants:
        pipeline = variant.config["pipeline"]
        assert pipeline["experiment"] == "sweep"
        assert pipeline["run"] == variant.strategy
        run = sweep.remote_run(variant, tmp_path / f"{variant.strategy}.yaml")
        # logs and the worker's session name come from the run's own folder
        assert run.output_dir == Path("/out/sweep") / variant.strategy
        assert run.log_file("h1") == run.output_dir / "logs" / "h1.log"
    sessions = {
        sweep.remote_run(v, tmp_path / f"{v.strategy}.yaml").session for v in variants
    }
    assert len(sessions) == 2


def test_the_sweep_needs_an_experiment_name():
    pipeline = {k: v for k, v in BASE["pipeline"].items() if k != "experiment"}
    with pytest.raises(ValueError, match="pipeline.experiment"):
        sweep.plan_variants({**BASE, "pipeline": pipeline}, ["entropy"], 1024)


def test_sweep_workers_check_for_each_other_often():
    for variant in sweep.plan_variants(BASE, ["entropy"], 1024):
        assert variant.config["pipeline"]["coordination"]["poll_interval_s"] == 5.0
    slow = {
        **BASE,
        "pipeline": {**BASE["pipeline"], "coordination": {"poll_interval_s": 60}},
    }
    variant = sweep.plan_variants(slow, ["entropy"], 1024)[0]
    assert variant.config["pipeline"]["coordination"]["poll_interval_s"] == 60, (
        "an explicit setting is kept"
    )


def test_spare_hosts_help_the_most_expensive_runs_in_turn():
    variants = sweep.plan_variants(
        BASE, ["first_n", "conch_kmeans", "texture"], 1024, ["h1", "h2", "h3"]
    )
    started = []

    def starter(variant, host):
        started.append((variant.strategy, host))
        return "started"

    helpers = sweep.add_helpers(variants, ["s1", "s2", "s3", "s4", "s5"], starter)
    # heaviest first: conch_kmeans, texture, first_n, then around again
    assert started == [
        ("conch_kmeans", "s1"),
        ("texture", "s2"),
        ("first_n", "s3"),
        ("conch_kmeans", "s4"),
        ("texture", "s5"),
    ]
    assert helpers == {
        "conch_kmeans": ["s1", "s4"],
        "texture": ["s2", "s5"],
        "first_n": ["s3"],
    }


def test_a_failing_spare_is_skipped_and_the_turn_is_not_lost():
    variants = sweep.plan_variants(
        BASE, ["conch_kmeans", "texture"], 1024, ["h1", "h2"]
    )
    reported = []
    helpers = sweep.add_helpers(
        variants,
        ["bad", "ok1", "ok2"],
        lambda variant, host: "failed: down" if host == "bad" else "started",
        lambda host, outcome: reported.append(host),
    )
    assert reported == ["bad"]
    assert helpers == {"conch_kmeans": ["ok1"], "texture": ["ok2"]}


def test_no_spares_or_no_runs_means_no_helpers():
    variants = sweep.plan_variants(BASE, ["entropy"], 1024, ["h1"])
    assert sweep.add_helpers(variants, [], lambda v, h: "started") == {"entropy": []}
    assert sweep.add_helpers([], ["s1"], lambda v, h: "started") == {}


def test_helpers_are_recorded_in_the_config_for_status_and_stop(tmp_path):
    variant = sweep.plan_variants(BASE, ["texture"], 1024, ["h1"])[0]
    helped = sweep.with_helpers(variant, ["s1", "s2"])
    assert helped.hosts == ["h1", "s1", "s2"]
    sweep.write_configs([helped], tmp_path)
    loaded = sweep.load_variants(tmp_path, ["texture"])[0]
    assert (loaded.host, loaded.helpers) == ("h1", ("s1", "s2"))
    assert loaded.config["pipeline"]["workers"]["hosts"] == ["h1", "s1", "s2"]
    # moving the run to another host drops the old helpers
    assert sweep.with_host(helped, "h9").hosts == ["h9"]
