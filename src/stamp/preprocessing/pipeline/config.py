"""Configuration for the patient-level preprocessing pipeline (`pipeline:` section).

Tiling and extractor settings stay in STAMP's own PreprocessingConfig
(`preprocessing:` section); this adds what the patient-level stages need.
"""

import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stamp.distributed.config import CoordinationConfig, WorkersConfig
from stamp.encoding.config import EncoderName


class Order(StrEnum):
    SAMPLE_THEN_EXTRACT = "sample_then_extract"
    """tile -> combine -> sample -> extract. Only sampled tiles go through the model."""

    EXTRACT_THEN_SAMPLE = "extract_then_sample"
    """tile -> extract_slides -> combine -> sample. Every tile goes through the
    model."""


class StepName(StrEnum):
    TILE = "tile"
    """WSI -> per-slide tile positions (+ STAMP tile cache)"""

    IMPORT_SLIDES = "import_slides"
    """per-slide features from plain `stamp preprocess` -> per-slide bags"""

    EXTRACT_SLIDES = "extract_slides"
    """per-slide tile positions -> per-slide features"""

    COMBINE = "combine"
    """per-slide bags -> one bag per patient"""

    SAMPLE = "sample"
    """patient bag -> patient bag with at most n_tiles tiles"""

    EXTRACT = "extract"
    """patient tile positions -> patient features"""

    ENCODE = "encode"
    """patient features -> one embedding per patient (e.g. TITAN); with
    encoding.slide_aggregation, one embedding per slide"""

    AGGREGATE = "aggregate"
    """a patient's slide embeddings -> one embedding (mean, max or concatenation)"""


ENCODING_STEPS = (StepName.ENCODE, StepName.AGGREGATE)
"""Steps after the training bags; their outputs are embeddings."""

SlideAggregation = Literal["mean", "max", "concat"]


SamplingStrategy = Literal[
    "first_n",
    "random",
    "tissue",
    "spatial",
    "entropy",
    "texture",
    "yottixel",
    "conch_kmeans",
]

STRATEGY_INPUT: dict[SamplingStrategy, Literal["positions", "images", "features"]] = {
    "first_n": "positions",
    "random": "positions",
    "spatial": "positions",
    "tissue": "images",
    "entropy": "images",
    "texture": "images",
    "yottixel": "images",
    "conch_kmeans": "features",
}
"""What each strategy looks at, which decides where it can run (see
`check_sampling`): positions work anywhere; images need the tile cache (so not
with imported features); features exist only in extract_then_sample."""
PadMode = Literal["none", "zero", "repeat"]


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    strategy: SamplingStrategy = "first_n"
    """first_n: first tiles of each slide in raster order (top-to-bottom, then
    left-to-right).
    random: uniformly random tiles within each slide (uses `seed`).
    tissue: the tiles with the least background within each slide (see
    `background_cutoff`); ties are broken randomly (uses `seed`). Reads the
    tile images from the tile cache, so it needs the tile step.
    spatial: divide the tissue into a grid (`spatial_cell_tiles`) and sample from
    each cell in proportion to its tile count (uses `seed`).
    entropy: the tiles with the highest grey-level (Shannon) entropy.
    texture: k-means on GLCM texture statistics, keeping the tile nearest each
    cluster centre (uses `seed`).
    yottixel: RGB-histogram k-means (`color_clusters`), then spatial k-means
    within each colour cluster (uses `seed`).
    conch_kmeans: k-means on the tile features, keeping the tile nearest each
    centre (uses `seed`). Needs conch1_5 features, so only extract_then_sample.
    The quota is the number of clusters for texture and conch_kmeans."""

    n_tiles: int = Field(default=8192, gt=0)
    """Bag size per patient. Each slide gets n_tiles // n_slides tiles."""

    redistribute: bool = True
    """If a slide has fewer tiles than its quota, give the unused quota to the
    patient's other slides. If False, such patients just get a smaller bag."""

    pad_mode: PadMode = "none"
    """What to do when a patient has fewer than n_tiles tiles in total.
    none: keep the smaller bag. zero: append all-zero tiles (mask=False).
    repeat: cycle through the patient's tiles again (mask=True)."""

    seed: int = 0
    """Recorded in the run's config.yaml; a run folder that already holds results
    made with another seed is rejected (give the run another name)."""

    background_cutoff: int = Field(default=220, gt=0, lt=256)
    """tissue only: a pixel counts as background if its grey value (0-255) is at
    least this. A tile's score is its fraction of non-background pixels."""

    spatial_cell_tiles: int = Field(default=8, gt=0)
    """spatial only: side of a grid cell, in tiles."""

    color_clusters: int = Field(default=9, gt=0)
    """yottixel only: number of RGB-histogram clusters (Yottixel uses 9)."""

    @property
    def tag(self) -> str:
        """Folder name for bags produced with these settings, e.g. first_n-8192."""
        if not self.enabled:
            return "full"
        parts = [self.strategy, str(self.n_tiles)]
        if self.strategy == "spatial":
            parts.append(f"cell{self.spatial_cell_tiles}")
        if self.strategy == "yottixel" and self.color_clusters != 9:
            parts.append(f"colors{self.color_clusters}")
        if self.strategy == "tissue" and self.background_cutoff != 220:
            parts.append(f"bg{self.background_cutoff}")
        if not self.redistribute:
            parts.append("noredist")
        if self.pad_mode != "none":
            parts.append(f"pad_{self.pad_mode}")
        return "-".join(parts)


class EncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int | None = Field(default=None, gt=0)
    """None: encode the patient's whole bag in one go.
    Otherwise: split a patient with more than chunk_size tiles into random chunks
    of at most that size, encode each chunk, and average the chunk embeddings.
    The result has the encoder's fixed size however many tiles the patient has,
    so every tile can be used (set sampling.enabled: false)."""

    seed: int = 0
    """Which tiles go into which chunk."""

    slide_aggregation: SlideAggregation | None = None
    """None: a patient's slides are encoded together, as one virtual slide.
    Otherwise each slide is encoded on its own (chunked within the slide if
    chunk_size is set), and the patient's slide embeddings are combined by
    mean: the average; max: element-wise maximum; concat: one after the other
    (slide order = sorted slide ids), zero-padded to `max_slides` slides.
    The per-slide embeddings are kept, so trying another aggregation only
    reruns the cheap `aggregate` step."""

    max_slides: int | None = Field(default=None, gt=0)
    """concat only: the embedding has max_slides * (encoder size) numbers.
    Must be at least the largest number of slides of any patient."""

    @model_validator(mode="after")
    def _check(self) -> "EncodingConfig":
        if (self.slide_aggregation == "concat") != (self.max_slides is not None):
            raise ValueError(
                "encoding.max_slides is needed for, and only for, "
                "slide_aggregation: concat"
            )
        return self

    @property
    def tag(self) -> str | None:
        """Folder name for embeddings made with these chunk settings, None if whole."""
        if self.chunk_size is None:
            return None
        return f"chunk{self.chunk_size}-seed{self.seed}"

    @property
    def aggregation_tag(self) -> str | None:
        if self.slide_aggregation is None:
            return None
        if self.slide_aggregation == "concat":
            return f"slides_concat{self.max_slides}"
        return f"slides_{self.slide_aggregation}"


class ClinicalColumns(BaseModel):
    """Accepted column names for each field of the clinical table, in priority order."""

    model_config = ConfigDict(extra="forbid")

    patient_id: list[str] = ["patient_id", "Patient"]
    slide_id: list[str] = ["slide_id", "StudyID/SlideID"]
    label: list[str] = ["label", "isLNM"]
    hospital: list[str] = ["hospital"]


class PipelineConfig(BaseModel):
    """Settings for `stamp preprocess` when run as the patient-level pipeline.

    Tiling and extractor settings (wsi_dir, output_dir, tile sizes, extractor,
    device, cache_dir, cutoffs, ...) come from the `preprocessing` section.
    """

    model_config = ConfigDict(extra="forbid")

    experiment: str
    """Name of the experiment (a question you answer with one or more runs). Its
    results go in `<output_dir>/<experiment>/`: a `shared/` folder for the work all
    its runs reuse (tiling, combining, slide features), and one folder per run."""

    run: str | None = None
    """Name of this run's folder inside the experiment. Default: derived from the
    extractor and sampling settings (e.g. `conch1_5-first_n-8192`). A sweep gives
    every run its own name."""

    clinical_table: Path
    """CSV with patient_id, slide_id, label, hospital (or the aliases in `columns`)."""

    columns: ClinicalColumns = ClinicalColumns()

    order: Order = Order.SAMPLE_THEN_EXTRACT

    steps: list[StepName] | None = None
    """Steps to run. Default: every step for `order` (see `default_steps`).
    Overridden on the command line by `--steps`."""

    sampling: SamplingConfig = SamplingConfig()

    slide_feat_dir: Path | None = None
    """extract_then_sample only: per-slide features from plain `stamp preprocess`
    (e.g. .../conch1_5-c1ab6684). Replaces the tile and extract_slides steps with
    import_slides."""

    encoder: EncoderName | None = None
    """Optional slide encoder (e.g. titan) run on each patient's bag."""

    encoding: EncodingConfig = EncodingConfig()

    encoder_device: str | None = None
    """Device for the encoder. Default: preprocessing.device."""

    batch_size: int = Field(default=64, gt=0)

    num_workers: int = Field(default=4, ge=0)
    """DataLoader workers decoding cached JPEG tiles during extraction."""

    allow_missing_slides: bool = False
    """If False, a patient with a missing or failed slide is an error.
    If True, the patient is combined from the slides that are available."""

    keep_intermediates: bool = True
    """Keep every stage's files. This is what lets other runs in the same
    output_dir reuse them (other sampling settings, encoders or extractors, or
    the other order). If False, a stage's files are deleted once the next stage
    has written its own, so only the latest finished stage remains (plus the
    training features and embeddings): saves disk, but a run with different
    settings starts again from the slides. Slide features of
    extract_then_sample are the biggest files."""

    drop_edge_tiles: bool = True
    """Drop tiles that extend past the bottom/right edge of the scan. STAMP's tiler
    can keep these (they are partly black padding)."""

    virtual_gap_tiles: int = Field(default=1, ge=0)
    """Empty tiles between slides when laying a patient's slides side by side."""

    coordination: CoordinationConfig = CoordinationConfig()

    workers: WorkersConfig | None = None
    """Hosts to run on with `stamp workers start`. Not needed to run locally."""

    @model_validator(mode="after")
    def _check_names(self) -> "PipelineConfig":
        for field, name in (("experiment", self.experiment), ("run", self.run)):
            if name is None:
                continue
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise ValueError(
                    f"pipeline.{field} {name!r}: use letters, digits, _ - and . only"
                )
            if name == "shared":
                raise ValueError(f"pipeline.{field} can't be 'shared' (reserved)")
        return self

    @model_validator(mode="after")
    def _check(self) -> "PipelineConfig":
        if self.slide_feat_dir is not None and self.order != Order.EXTRACT_THEN_SAMPLE:
            raise ValueError(
                "slide_feat_dir only applies to order: extract_then_sample"
            )
        check_sampling(self)
        if self.encoder is None and (
            self.encoding.chunk_size is not None
            or self.encoding.slide_aggregation is not None
        ):
            raise ValueError(
                "encoding.chunk_size and encoding.slide_aggregation need "
                "pipeline.encoder"
            )
        if self.steps is not None:
            check_steps(self, self.steps)
        return self

    def default_steps(self) -> list[StepName]:
        """All steps that make sense for this configuration, in run order."""
        if self.order == Order.SAMPLE_THEN_EXTRACT:
            steps = [StepName.TILE, StepName.COMBINE, StepName.SAMPLE, StepName.EXTRACT]
        elif self.slide_feat_dir is not None:
            steps = [StepName.IMPORT_SLIDES, StepName.COMBINE, StepName.SAMPLE]
        else:
            steps = [
                StepName.TILE,
                StepName.EXTRACT_SLIDES,
                StepName.COMBINE,
                StepName.SAMPLE,
            ]
        if not self.sampling.enabled:
            steps.remove(StepName.SAMPLE)
        if self.encoder is not None:
            steps.append(StepName.ENCODE)
            if self.encoding.slide_aggregation is not None:
                steps.append(StepName.AGGREGATE)
        return steps


def check_steps(cfg: PipelineConfig, steps: Sequence[StepName]) -> None:
    """Reject steps that don't belong to the configured order, before anything runs."""
    allowed = set(cfg.default_steps())
    if cfg.encoder is None and (
        step := next((s for s in steps if s in ENCODING_STEPS), None)
    ):
        raise ValueError(f"step '{step}' needs pipeline.encoder (e.g. titan)")
    if StepName.AGGREGATE in steps and cfg.encoding.slide_aggregation is None:
        raise ValueError("step 'aggregate' needs pipeline.encoding.slide_aggregation")
    if StepName.SAMPLE in steps and not cfg.sampling.enabled:
        raise ValueError("step 'sample' needs pipeline.sampling.enabled: true")
    if bad := [str(s) for s in steps if s not in allowed]:
        raise ValueError(
            f"steps {bad} don't apply to order {cfg.order}"
            f"{' with slide_feat_dir' if cfg.slide_feat_dir else ''}; "
            f"valid steps: {[str(s) for s in cfg.default_steps()]}"
        )


def check_sampling(cfg: PipelineConfig) -> None:
    """Reject a sampling strategy that can't work where this configuration
    samples: before anything runs, with what to change."""
    if not cfg.sampling.enabled:
        return
    strategy = cfg.sampling.strategy
    needs = STRATEGY_INPUT[strategy]
    if needs == "features" and cfg.order != Order.EXTRACT_THEN_SAMPLE:
        raise ValueError(
            f"sampling strategy '{strategy}' uses the tile features, which only "
            f"exist before sampling in order: extract_then_sample (not {cfg.order})"
        )
    if needs == "images" and cfg.slide_feat_dir is not None:
        raise ValueError(
            f"sampling strategy '{strategy}' uses the tile images, but "
            "slide_feat_dir imports features without a tile cache; use a "
            "strategy that only needs positions (first_n, random, spatial)"
        )
