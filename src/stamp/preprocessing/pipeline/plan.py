"""Which files each step reads and writes: the run's stages and their folders.

This is the only module that knows how the steps connect. The steps take input
paths and an output path, and the scheduler only sees `Step` objects.

Stage folders are named after the settings that shaped their contents, so one
`output_dir` holds any number of runs (other extractors, sampling settings,
orders, encoders) and runs share whatever they have in common. With T the tiling
settings (e.g. `256um_224px_3fa91c`), G = `T_gap1` (the combine settings),
E the extractor and S the sampling settings (e.g. `first_n-8192`, or `full`):

    tile_positions/T/<slide>.h5            tile positions per slide
    patient_positions/G/<patient>.h5       all tile positions of a patient
    sampled_positions/G/S/<patient>.h5     the sampled tile positions
    features/E/G/S/<patient>.h5            sampled tiles with features  <- training
    embeddings/<encoder>/E/G/S/<patient>.h5   one embedding per patient

extract_then_sample uses `slide_features/E/T`, `patient_features/E/G` and
`sampled_features/E/G/S` instead of the positions folders, and ends in the
same `features/E/G/S` folder: both orders make the same training bags, so
whichever ran first has done the work for the other. Alongside: `overviews/`
(thumbnails, if save_visualisations), `logs/` (workers).

Files of earlier stages are kept (`keep_intermediates`, the default), which is
what lets a second run reuse them. Without it, a stage's files are deleted once
the next stage has written its own, so a run can't be repeated with other
sampling settings.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from stamp.distributed.workqueue import Step
from stamp.encoding.encoder import Encoder
from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.data.clinical import Patient, Slide
from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.pipeline.config import PipelineConfig, StepName
from stamp.preprocessing.steps import (
    combine,
    encode,
    extract,
    import_slides,
    sample,
    tile,
)
from stamp.preprocessing.steps.models import Lazy, load_encoder, load_extractor

SLIDE_STEPS = frozenset(
    {StepName.TILE, StepName.IMPORT_SLIDES, StepName.EXTRACT_SLIDES}
)
"""Steps that write one file per slide; the others write one file per patient."""

SETTINGS_FILE = "settings.json"
"""Written into each kept stage folder: the settings it was made with."""


@dataclass(frozen=True)
class Stage:
    step: StepName
    folder: Path
    per_slide: bool
    kept: bool
    """Never deleted by later stages (training features, embeddings)."""

    def file_of(self, item: Slide | Patient) -> Path:
        return self.folder / f"{item.file_stem}.h5"


@dataclass(frozen=True)
class Layout:
    root: Path
    stages: tuple[Stage, ...]
    tile_overviews: Path
    sampling_overviews: Path

    def stage(self, step: StepName) -> Stage:
        return next(stage for stage in self.stages if stage.step == step)

    @property
    def training_bags(self) -> Path:
        """The `features/...` folder: the last stage before encoding."""
        return [s for s in self.stages if s.step != StepName.ENCODE][-1].folder

    @property
    def embeddings(self) -> Path | None:
        encoding = [s for s in self.stages if s.step == StepName.ENCODE]
        return encoding[0].folder if encoding else None


def layout_for(pre: PreprocessingConfig, cfg: PipelineConfig) -> Layout:
    steps = cfg.default_steps()
    last_data_step = [s for s in steps if s != StepName.ENCODE][-1]
    extractor = str(pre.extractor)
    base = _base_tag(pre, cfg)
    patient = f"{base}_gap{cfg.virtual_gap_tiles}"
    sampling = cfg.sampling.tag

    stages = []
    has_feats = False
    for step in steps:
        has_feats |= step in (
            StepName.IMPORT_SLIDES,
            StepName.EXTRACT_SLIDES,
            StepName.EXTRACT,
        )
        if step == StepName.ENCODE:
            relative = Path(
                "embeddings", str(cfg.encoder), extractor, patient, sampling
            )
        elif step == last_data_step:
            relative = Path("features", extractor, patient, sampling)
        else:
            relative = _folder(step, has_feats, extractor, base, patient, sampling)
        stages.append(
            Stage(
                step=step,
                folder=pre.output_dir / relative,
                per_slide=step in SLIDE_STEPS,
                kept=step in (last_data_step, StepName.ENCODE),
            )
        )
    return Layout(
        root=pre.output_dir,
        stages=tuple(stages),
        tile_overviews=pre.output_dir / "overviews" / "tiles" / base,
        sampling_overviews=pre.output_dir
        / "overviews"
        / "sampled"
        / patient
        / sampling,
    )


def _folder(
    step: StepName,
    has_feats: bool,
    extractor: str,
    base: str,
    patient: str,
    sampling: str,
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
        case StepName.SAMPLE if has_feats:
            return Path("sampled_features", extractor, patient, sampling)
        case StepName.SAMPLE:
            return Path("sampled_positions", patient, sampling)
    raise ValueError(f"no folder for step {step}")


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


def run_settings(pre: PreprocessingConfig, cfg: PipelineConfig) -> dict[str, Any]:
    """The settings that shape the results, as written to `settings.json`."""
    return {
        "order": str(cfg.order),
        "extractor": str(pre.extractor),
        "tile_size_um": float(pre.tile_size_um),
        "tile_size_px": int(pre.tile_size_px),
        "brightness_cutoff": pre.brightness_cutoff,
        "canny_cutoff": pre.canny_cutoff,
        "drop_edge_tiles": cfg.drop_edge_tiles,
        "virtual_gap_tiles": cfg.virtual_gap_tiles,
        "sampling": cfg.sampling.model_dump(mode="json"),
        "slide_feat_dir": None
        if cfg.slide_feat_dir is None
        else str(cfg.slide_feat_dir),
        "encoder": None if cfg.encoder is None else str(cfg.encoder),
    }


def record_run_settings(pre: PreprocessingConfig, cfg: PipelineConfig) -> None:
    """Write `settings.json` into the folders holding the run's results, so a
    folder says what made it. The folder names already keep runs apart."""
    text = json.dumps(run_settings(pre, cfg), indent=2) + "\n"
    for stage in layout_for(pre, cfg).stages:
        if not stage.kept:
            continue
        path = stage.folder / SETTINGS_FILE
        if path.exists() and path.read_text() == text:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(path)
