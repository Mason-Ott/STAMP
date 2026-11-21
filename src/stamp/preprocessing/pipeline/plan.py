"""Which files each step reads and writes: the run's stages and their folders.

This is the only module that knows how the steps connect. The steps take input
paths and an output path, and the scheduler only sees `Step` objects.

Results are organised by experiment and run:

    <output_dir>/<experiment>/
        shared/                      work every run of the experiment reuses
            tile_positions/T/            tile positions per slide
            patient_positions/G/         all tile positions of a patient
            slide_features/E/T/          extract_then_sample: per-slide features
            patient_features/E/G/        extract_then_sample: patient features
            overviews/tiles/T/           tile thumbnails
        <run>/                       one run (a sampling method, a model, ...)
            config.yaml  README.md       what it ran with, and a guide to the folder
            sampled_positions/           sample_then_extract: the sampled tiles
            sampled_features/            extract_then_sample: the sampled tiles
            features/                    training bags (sampled tiles with features)
            embeddings/<encoder>/        one embedding per patient (with an encoder)
            overviews/                   sampling thumbnails (save_visualisations)
            logs/                        worker logs

T = tiling settings (e.g. `256um_224px_3fa91c`), G = `T_gap1`, E = the extractor.
The shared folders are keyed by the settings that shape them, so runs of one
experiment can differ in tile size or extractor and still share what they can.
README.md in each run (see `describe.py`) draws the steps and lists the folders.

Unless keep_intermediates is false (the default is true), earlier stages' files are
kept, which is what lets runs reuse them. Without it, a stage's files are deleted
once the next stage has written its own, so another run can't reuse them.
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from stamp.distributed.fs import ensure_dir
from stamp.distributed.workqueue import Step
from stamp.encoding.encoder import Encoder
from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.data.clinical import Patient, Slide
from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.pipeline import describe
from stamp.preprocessing.pipeline.config import PipelineConfig, StepName
from stamp.preprocessing.steps import (
    combine,
    encode,
    extract,
    import_slides,
    sample,
    tile,
)
from stamp.preprocessing.steps.models import (
    Lazy,
    encoder_accepts,
    load_encoder,
    load_extractor,
)

SLIDE_STEPS = frozenset(
    {StepName.TILE, StepName.IMPORT_SLIDES, StepName.EXTRACT_SLIDES}
)
"""Steps that write one file per slide; the others write one file per patient."""

CONFIG_FILE = "config.yaml"
"""Written into the run folder: the configuration it ran with."""


@dataclass(frozen=True)
class Stage:
    step: StepName
    folder: Path
    per_slide: bool
    kept: bool
    """Never deleted by later stages (training features, embeddings)."""

    shared: bool = False
    """Lives in the experiment's shared folder (used by all its runs)."""

    def file_of(self, item: Slide | Patient) -> Path:
        return self.folder / f"{item.file_stem}.h5"


@dataclass(frozen=True)
class Layout:
    experiment_dir: Path
    run_dir: Path
    stages: tuple[Stage, ...]
    tile_overviews: Path
    sampling_overviews: Path

    @property
    def shared_dir(self) -> Path:
        return self.experiment_dir / "shared"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    def stage(self, step: StepName) -> Stage:
        return next(stage for stage in self.stages if stage.step == step)

    @property
    def training_bags(self) -> Path:
        """The run's `features` folder: the last stage before encoding."""
        return [s for s in self.stages if s.step != StepName.ENCODE][-1].folder

    @property
    def embeddings(self) -> Path | None:
        encoding = [s for s in self.stages if s.step == StepName.ENCODE]
        return encoding[0].folder if encoding else None


def run_name(pre: PreprocessingConfig, cfg: PipelineConfig) -> str:
    """The run's folder name: `pipeline.run`, or derived from the extractor and
    the sampling settings."""
    return cfg.run or f"{pre.extractor}-{cfg.sampling.tag}"


def encoding_possible(pre: PreprocessingConfig, cfg: PipelineConfig) -> bool:
    """Whether the encoder accepts the features the extractor makes."""
    return cfg.encoder is None or encoder_accepts(cfg.encoder, pre.extractor)


def active_steps(pre: PreprocessingConfig, cfg: PipelineConfig) -> list[StepName]:
    """`cfg.default_steps()`, without encoding if the encoder can't use the
    extractor's features (e.g. TITAN needs conch1_5)."""
    steps = cfg.default_steps()
    if encoding_possible(pre, cfg):
        return steps
    return [s for s in steps if s != StepName.ENCODE]


def layout_for(pre: PreprocessingConfig, cfg: PipelineConfig) -> Layout:
    steps = active_steps(pre, cfg)
    last_data_step = [s for s in steps if s != StepName.ENCODE][-1]
    extractor = str(pre.extractor)
    base = _base_tag(pre, cfg)
    patient = f"{base}_gap{cfg.virtual_gap_tiles}"

    experiment_dir = pre.output_dir / cfg.experiment
    run_dir = experiment_dir / run_name(pre, cfg)
    shared_dir = experiment_dir / "shared"

    stages = []
    has_feats = False
    for step in steps:
        has_feats |= step in (
            StepName.IMPORT_SLIDES,
            StepName.EXTRACT_SLIDES,
            StepName.EXTRACT,
        )
        shared = False
        if step == StepName.ENCODE:
            folder = run_dir / "embeddings" / str(cfg.encoder)
        elif step == last_data_step:
            folder = run_dir / "features"
        elif step == StepName.SAMPLE:
            folder = run_dir / (
                "sampled_features" if has_feats else "sampled_positions"
            )
        else:
            folder = shared_dir / _shared_folder(
                step, has_feats, extractor, base, patient
            )
            shared = True
        stages.append(
            Stage(
                step=step,
                folder=folder,
                per_slide=step in SLIDE_STEPS,
                kept=step in (last_data_step, StepName.ENCODE),
                shared=shared,
            )
        )
    return Layout(
        experiment_dir=experiment_dir,
        run_dir=run_dir,
        stages=tuple(stages),
        tile_overviews=shared_dir / "overviews" / "tiles" / base,
        sampling_overviews=run_dir / "overviews",
    )


def _shared_folder(
    step: StepName, has_feats: bool, extractor: str, base: str, patient: str
) -> Path:
    match step:
        case StepName.TILE:
            return Path("tile_positions", base)
        case StepName.IMPORT_SLIDES | StepName.EXTRACT_SLIDES:
            return Path("slide_features", extractor, base)
        case StepName.COMBINE if has_feats:
            return Path("patient_features", extractor, patient)
        case StepName.COMBINE:
            return Path("patient_positions", patient)
    raise ValueError(f"no shared folder for step {step}")


def _base_tag(pre: PreprocessingConfig, cfg: PipelineConfig) -> str:
    """Names the per-slide tiles: the tiling settings, or the imported features."""
    if cfg.slide_feat_dir is not None:
        digest = _digest(str(cfg.slide_feat_dir.resolve()), cfg.drop_edge_tiles)
        return f"imported_{digest}"
    digest = _digest(pre.brightness_cutoff, pre.canny_cutoff, cfg.drop_edge_tiles)
    return f"{pre.tile_size_um:g}um_{pre.tile_size_px}px_{digest}"


def _digest(*parts: object) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:6]


@dataclass(frozen=True)
class ModelLoaders:
    """How to get the models. Tests swap in small fakes here."""

    extractor: Lazy[Extractor]
    encoder: Lazy[Encoder]

    @classmethod
    def from_config(
        cls, pre: PreprocessingConfig, cfg: PipelineConfig
    ) -> "ModelLoaders":
        def missing_encoder() -> Encoder:
            raise ValueError("pipeline.encoder is not set")

        encoder_device = cfg.encoder_device or str(pre.device)
        return cls(
            extractor=Lazy(partial(load_extractor, pre.extractor, str(pre.device))),
            encoder=Lazy(
                partial(load_encoder, cfg.encoder, encoder_device)
                if cfg.encoder is not None
                else missing_encoder
            ),
        )


def build_plan(
    pre: PreprocessingConfig,
    cfg: PipelineConfig,
    patients: list[Patient],
    models: ModelLoaders,
) -> dict[StepName, Step]:
    """A `Step` for every stage of this configuration, wired to its folders."""
    layout = layout_for(pre, cfg)
    runners = _runners(pre, cfg, layout, models)
    slides = [s for p in patients for s in p.slides if s.path is not None]
    patient_of = {s.slide_id: p for p in patients for s in p.slides}

    def item_in(stage: Stage, item: Slide | Patient) -> Slide | Patient | None:
        """`item` as `stage` sees it: a slide stage has no single file per patient."""
        if stage.per_slide:
            return item if isinstance(item, Slide) else None
        return patient_of[item.slide_id] if isinstance(item, Slide) else item

    plan: dict[StepName, Step] = {}
    for index, stage in enumerate(layout.stages):
        previous = layout.stages[index - 1] if index > 0 else None
        # Later stages whose files show this item was consumed. Encoding never
        # deletes its input, so its output doesn't count.
        later = [s for s in layout.stages[index + 1 :] if s.step != StepName.ENCODE]

        def inputs_of(
            item: Slide | Patient,
            stage: Stage = stage,
            previous: Stage | None = previous,
        ) -> list[Path]:
            if stage.step == StepName.IMPORT_SLIDES and isinstance(item, Slide):
                return [
                    import_slides.stamp_feature_file(item, _import_settings(pre, cfg))
                ]
            if previous is None:
                return []
            if previous.per_slide and isinstance(item, Patient):
                return [previous.file_of(s) for s in item.slides]
            return [previous.file_of(item)]

        def later_outputs_of(
            item: Slide | Patient, later: list[Stage] = later
        ) -> list[Path]:
            return [s.file_of(i) for s in later if (i := item_in(s, item)) is not None]

        plan[stage.step] = Step(
            name=str(stage.step),
            items=slides if stage.per_slide else patients,
            output_of=stage.file_of,
            inputs_of=inputs_of,
            run=runners[stage.step],
            allow_missing_inputs=(
                stage.step == StepName.COMBINE and cfg.allow_missing_slides
            ),
            later_outputs_of=later_outputs_of,
            delete_inputs_when_done=(
                not cfg.keep_intermediates
                and previous is not None
                and not previous.kept
                and stage.step != StepName.IMPORT_SLIDES  # inputs are the user's files
            ),
        )
    return plan


def _runners(
    pre: PreprocessingConfig, cfg: PipelineConfig, layout: Layout, models: ModelLoaders
) -> dict[StepName, Callable[..., None]]:
    """The function that does each step's work, with its settings bound."""
    extraction = extract.ExtractionSettings(
        device=str(pre.device), batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    runners: dict[StepName, Callable[..., None]] = {
        StepName.TILE: partial(
            tile.run,
            settings=tile.TilingSettings.from_config(
                pre,
                cache_dir=pre.cache_dir or pre.output_dir / "tile_cache",
                drop_edge_tiles=cfg.drop_edge_tiles,
                overview_dir=layout.tile_overviews,
            ),
        ),
        StepName.EXTRACT_SLIDES: partial(
            extract.run, extractor=models.extractor, settings=extraction
        ),
        StepName.COMBINE: partial(
            combine.run,
            settings=combine.CombineSettings(gap_tiles=cfg.virtual_gap_tiles),
        ),
        StepName.SAMPLE: partial(
            sample.run,
            settings=cfg.sampling,
            overview_dir=layout.sampling_overviews if pre.save_visualisations else None,
            overview_max_dimension=pre.visualisation_max_dimension,
        ),
        StepName.EXTRACT: partial(
            extract.run, extractor=models.extractor, settings=extraction
        ),
        StepName.ENCODE: partial(
            encode.run,
            encoder=models.encoder,
            settings=encode.EncodingSettings(
                device=cfg.encoder_device or str(pre.device)
            ),
        ),
    }
    if cfg.slide_feat_dir is not None:
        runners[StepName.IMPORT_SLIDES] = partial(
            import_slides.run, settings=_import_settings(pre, cfg)
        )
    return runners


def _import_settings(
    pre: PreprocessingConfig, cfg: PipelineConfig
) -> import_slides.ImportSettings:
    if cfg.slide_feat_dir is None:
        raise ValueError("import_slides needs pipeline.slide_feat_dir")
    return import_slides.ImportSettings(
        feat_dir=cfg.slide_feat_dir,
        wsi_dir=pre.wsi_dir,
        default_slide_mpp=pre.default_slide_mpp,
        drop_edge_tiles=cfg.drop_edge_tiles,
    )


def run_config(pre: PreprocessingConfig, cfg: PipelineConfig) -> dict[str, Any]:
    """The configuration a run used, as written to `config.yaml`. Leaves out what
    doesn't change the results (hosts, locking, which steps were asked for)."""
    return {
        "preprocessing": pre.model_dump(mode="json"),
        "pipeline": cfg.model_dump(
            mode="json", exclude={"workers", "coordination", "steps"}
        ),
    }


def prepare_run(pre: PreprocessingConfig, cfg: PipelineConfig) -> Layout:
    """Create the run's folder with its `config.yaml`, `README.md`, and the
    experiment's index, before any step runs.

    A run folder is for one set of results. If it already holds a config with
    different result-shaping settings (another seed, tile size, extractor, ...),
    stop: the new results would be mixed up with, or mistaken for, the old ones.
    """
    layout = layout_for(pre, cfg)
    config = run_config(pre, cfg)
    path = layout.run_dir / CONFIG_FILE
    if path.exists():
        _check_same_run(layout.run_dir, yaml.safe_load(path.read_text()), config)

    ensure_dir(layout.run_dir)
    _write_if_changed(
        path,
        "# The configuration this run was made with.\n"
        + yaml.safe_dump(config, sort_keys=False),
    )
    _write_if_changed(
        layout.run_dir / "README.md", describe.describe_run(pre, cfg, layout)
    )
    _write_if_changed(
        layout.experiment_dir / "README.md",
        describe.describe_experiment(layout.experiment_dir),
    )
    return layout


def _write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _identity(config: dict[str, Any]) -> dict[str, Any]:
    """The settings of a config that shape a run's results (not, e.g., the
    encoder: embeddings of several encoders can sit in one run)."""
    pre, pipeline = config["preprocessing"], config["pipeline"]
    return {
        "sampling": pipeline["sampling"],
        "order": pipeline["order"],
        "drop_edge_tiles": pipeline["drop_edge_tiles"],
        "virtual_gap_tiles": pipeline["virtual_gap_tiles"],
        "slide_feat_dir": pipeline["slide_feat_dir"],
        **{
            key: pre[key]
            for key in (
                "extractor",
                "tile_size_um",
                "tile_size_px",
                "brightness_cutoff",
                "canny_cutoff",
            )
        },
    }


def _check_same_run(
    run_dir: Path, recorded: dict[str, Any], current: dict[str, Any]
) -> None:
    old, new = _identity(recorded), _identity(current)
    changes = []
    for key in old:
        if old[key] == new[key]:
            continue
        if isinstance(old[key], dict):
            changes += [
                f"{key}.{k}: {old[key].get(k)!r} -> {new[key].get(k)!r}"
                for k in sorted(set(old[key]) | set(new[key]))
                if old[key].get(k) != new[key].get(k)
            ]
        else:
            changes.append(f"{key}: {old[key]!r} -> {new[key]!r}")
    if changes:
        raise ValueError(
            f"{run_dir} already holds results made with different settings "
            f"({'; '.join(changes)}). Give this run another name (pipeline.run) "
            "or experiment."
        )
