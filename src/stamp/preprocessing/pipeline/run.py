"""Entry points: run the pipeline's steps, or report how far they have got."""

import logging
import random
import time
from collections.abc import Sequence

from stamp.distributed.progress import StepProgress, step_progress
from stamp.distributed.worker import run_as_worker
from stamp.distributed.workqueue import StepReport, run_step
from stamp.preprocessing.config import ExtractorName, PreprocessingConfig
from stamp.preprocessing.data.clinical import load_patients
from stamp.preprocessing.pipeline.config import (
    ENCODING_STEPS,
    PipelineConfig,
    StepName,
    check_steps,
)
from stamp.preprocessing.pipeline.plan import (
    ModelLoaders,
    active_steps,
    build_plan,
    encoding_possible,
    layout_for,
    prepare_run,
)

_logger = logging.getLogger("stamp")


def run_pipeline_(
    pre: PreprocessingConfig,
    cfg: PipelineConfig,
    steps: Sequence[StepName] | None = None,
    *,
    models: ModelLoaders | None = None,
) -> dict[StepName, StepReport]:
    """Run `steps` (default: every step for the configured order) one after another.

    A failing item doesn't stop the run: it gets a .failed marker, later steps
    report its outputs as missing, and the reports returned here count failures.
    """
    steps = list(steps) if steps is not None else active_steps(pre, cfg)
    check_steps(cfg, steps)
    if not encoding_possible(pre, cfg):
        if any(s in ENCODING_STEPS for s in steps):
            raise ValueError(
                f"encoder {cfg.encoder} can't use features from {pre.extractor}"
            )
        _logger.warning(
            f"pipeline: skipping encoding: {cfg.encoder} can't use features "
            f"from {pre.extractor}"
        )
    if (
        cfg.sampling.enabled
        and cfg.sampling.strategy == "conch_kmeans"
        and pre.extractor != ExtractorName.CONCH1_5
    ):
        raise ValueError(
            "sampling strategy 'conch_kmeans' clusters conch1_5 features, not "
            f"{pre.extractor} features; set preprocessing.extractor: conch1_5"
        )
    patients = load_patients(cfg.clinical_table, cfg.columns, pre.wsi_dir)
    max_slides = max((len(p.slides) for p in patients), default=0)
    _logger.info(f"pipeline: at most {max_slides} slides per patient")
    if (limit := cfg.encoding.max_slides) is not None and max_slides > limit:
        raise ValueError(
            f"a patient has {max_slides} slides, more than encoding.max_slides "
            f"= {limit}"
        )
    prepare_run(pre, cfg)
    plan = build_plan(pre, cfg, patients, models or ModelLoaders.from_config(pre, cfg))
    _logger.info(
        f"pipeline: {len(patients)} patients, order {cfg.order}, "
        f"sampling {cfg.sampling.tag}, steps {[str(s) for s in steps]}"
    )

    reports = {}
    for name in steps:
        reports[name] = run_step(plan[name], cfg.coordination)

    layout = layout_for(pre, cfg)
    _logger.info(f"pipeline: training bags are in {layout.training_bags}")
    if layout.embeddings is not None:
        _logger.info(f"pipeline: embeddings are in {layout.embeddings}")
    return reports


def run_pipeline_worker(
    pre: PreprocessingConfig, cfg: PipelineConfig, steps: Sequence[StepName] | None
) -> int:
    """`stamp worker`: the pipeline as one of several workers. Returns an exit code."""

    def job() -> bool:
        # Workers started together would otherwise all create the same folders and
        # claim the same first item in the same moment.
        time.sleep(random.uniform(0, 3))
        reports = run_pipeline_(pre, cfg, steps)
        return all(report.ok for report in reports.values())

    return run_as_worker(job, cfg.workers)


def pipeline_progress(
    pre: PreprocessingConfig, cfg: PipelineConfig
) -> dict[StepName, StepProgress]:
    """Progress of every step this configuration runs, in order, from its outputs."""
    patients = load_patients(cfg.clinical_table, cfg.columns, pre.wsi_dir)
    plan = build_plan(pre, cfg, patients, ModelLoaders.from_config(pre, cfg))
    stale_after_s = cfg.coordination.lock_stale_after_s
    return {
        name: step_progress(plan[name], stale_after_s)
        for name in active_steps(pre, cfg)
    }
