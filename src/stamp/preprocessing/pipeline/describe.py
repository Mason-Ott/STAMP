"""Plain-text guides written into a run and an experiment folder.

A run's README.md draws the steps in order (with the settings each one uses and
where its output goes) and lists the run's folders. The experiment's README.md
lists the runs it holds. Both are generated from the configuration, so they can't
drift from what the pipeline does.
"""

import os
from pathlib import Path
from typing import Any

import yaml

from stamp.preprocessing.config import PreprocessingConfig
from stamp.preprocessing.pipeline.config import (
    STRATEGY_INPUT,
    PipelineConfig,
    StepName,
)

STEP_SUMMARY = {
    StepName.TILE: "WSI -> tile positions, one file per slide",
    StepName.IMPORT_SLIDES: "plain `stamp preprocess` features -> one file per slide",
    StepName.EXTRACT_SLIDES: "tile positions -> tile features, one file per slide",
    StepName.COMBINE: "slides -> one bag per patient (slides laid side by side)",
    StepName.SAMPLE: "patient bag -> at most n_tiles tiles",
    StepName.EXTRACT: "sampled tile positions -> tile features",
    StepName.ENCODE: "patient features -> one embedding per patient",
}

STRATEGY_SUMMARY = {
    "first_n": "the first tiles of each slide in raster order",
    "random": "uniformly random tiles from each slide",
    "tissue": "the tiles with the least background",
    "spatial": "random tiles from each cell of a grid, in proportion to its tile count",
    "entropy": "the tiles with the highest grey-level entropy",
    "texture": "k-means on GLCM texture features, one tile per cluster",
    "yottixel": "RGB-histogram k-means, then spatial k-means in each colour cluster",
    "conch_kmeans": "k-means on the tile features, one tile per cluster",
}


# `layout` is plan.Layout; typed loosely because plan.py imports this module.


def describe_run(pre: PreprocessingConfig, cfg: PipelineConfig, layout: Any) -> str:
    from stamp.preprocessing.pipeline.plan import run_name

    sampling = cfg.sampling
    lines = [
        f"# Run `{run_name(pre, cfg)}` of experiment `{cfg.experiment}`",
        "",
        f"- sampling: {_sampling_text(cfg)}",
        f"- extractor: {pre.extractor}"
        + (f", encoder: {cfg.encoder}" if cfg.encoder else ", no encoder"),
        f"- order: {cfg.order} "
        + (
            "(sample first, then extract features of the sampled tiles)"
            if str(cfg.order) == "sample_then_extract"
            else "(extract features of every tile, then sample)"
        ),
        "",
        "Everything this run was made with is in [config.yaml](config.yaml).",
        "Run it again with `stamp -c config.yaml preprocess` "
        "(finished work is skipped).",
        "",
        "## Order of steps",
        "",
        "```",
    ]
    steps = layout.stages
    lines.append("  " + "  -->  ".join(str(stage.step) for stage in steps))
    lines.append("")
    for number, stage in enumerate(steps, start=1):
        where = os.path.relpath(stage.folder, layout.run_dir)
        note = "   (shared by the runs of this experiment)" if stage.shared else ""
        lines += [
            f" {number}. {stage.step}: {STEP_SUMMARY[stage.step]}",
            f"      settings: {_step_settings(stage.step, pre, cfg)}",
            f"      output:   {where}/{note}",
        ]
        if number < len(steps):
            lines += ["         |", "         v"]
    lines += ["```", ""]

    if sampling.enabled and StepName.SAMPLE in [s.step for s in steps]:
        kind = {
            "positions": "tile positions only",
            "images": "the tile images",
            "features": "the tile features",
        }[STRATEGY_INPUT[sampling.strategy]]
        lines += [
            f"Sampling `{sampling.strategy}` ({STRATEGY_SUMMARY[sampling.strategy]}) "
            f"looks at {kind}.",
            "",
        ]

    lines += ["## Where things are", "", "```", *_tree(pre, cfg, layout), "```", ""]
    bags = os.path.relpath(layout.training_bags, layout.run_dir)
    summary = f"The training bags are in `{bags}/`"
    if layout.embeddings:
        embeddings = os.path.relpath(layout.embeddings, layout.run_dir)
        summary += f", the embeddings in `{embeddings}/`"
    lines += [summary + ".", ""]
    return "\n".join(lines)


def describe_experiment(experiment_dir: Path) -> str:
    """README for the experiment folder: its runs, from their config.yaml files."""
    rows = []
    for config_file in sorted(experiment_dir.glob("*/config.yaml")):
        if config_file.parent.name == "shared":
            continue
        try:
            config = yaml.safe_load(config_file.read_text())
            sampling = config["pipeline"]["sampling"]
            sampler = sampling["strategy"] if sampling["enabled"] else "none"
            rows.append(
                f"| [{config_file.parent.name}]({config_file.parent.name}/README.md) "
                f"| {sampler} "
                f"| {sampling['n_tiles'] if sampling['enabled'] else 'all'} "
                f"| {sampling['seed']} | {config['preprocessing']['extractor']} "
                f"| {config['pipeline']['encoder'] or '-'} "
                f"| {config['pipeline']['order']} |"
            )
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            rows.append(
                f"| {config_file.parent.name} | (config unreadable) | | | | | |"
            )
    return "\n".join(
        [
            f"# Experiment `{experiment_dir.name}`",
            "",
            "One folder per run, each with its `config.yaml` and a `README.md` that "
            "explains the order of steps and where its outputs are. `shared/` holds "
            "the work all runs reuse (tile positions, combined patients, slide "
            "features).",
            "",
            "| run | sampling | n_tiles | seed | extractor | encoder | order |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )


def _sampling_text(cfg: PipelineConfig) -> str:
    sampling = cfg.sampling
    if not sampling.enabled:
        return "none (every tile)"
    return (
        f"{sampling.strategy}, {sampling.n_tiles} tiles per patient, "
        f"seed {sampling.seed}, pad_mode {sampling.pad_mode}"
    )


def _step_settings(
    step: StepName, pre: PreprocessingConfig, cfg: PipelineConfig
) -> str:
    sampling = cfg.sampling
    match step:
        case StepName.TILE:
            return (
                f"tiles {pre.tile_size_um:g} um / {pre.tile_size_px} px, "
                f"brightness_cutoff {pre.brightness_cutoff}, "
                f"canny_cutoff {pre.canny_cutoff}, "
                f"drop_edge_tiles {cfg.drop_edge_tiles}"
            )
        case StepName.IMPORT_SLIDES:
            return f"slide_feat_dir {cfg.slide_feat_dir}"
        case StepName.EXTRACT_SLIDES | StepName.EXTRACT:
            return f"extractor {pre.extractor}, batch_size {cfg.batch_size}"
        case StepName.COMBINE:
            return f"virtual_gap_tiles {cfg.virtual_gap_tiles}"
        case StepName.SAMPLE:
            return (
                f"strategy {sampling.strategy}, n_tiles {sampling.n_tiles}, "
                f"seed {sampling.seed}, redistribute {sampling.redistribute}, "
                f"pad_mode {sampling.pad_mode}"
            )
        case StepName.ENCODE:
            return f"encoder {cfg.encoder}"
    return ""


def _tree(pre: PreprocessingConfig, cfg: PipelineConfig, layout: Any) -> list[str]:
    from stamp.preprocessing.pipeline.plan import CONFIG_FILE

    experiment = layout.experiment_dir
    run = layout.run_dir.relative_to(experiment)
    entries: list[tuple[Path, str, bool]] = [  # (path, note, is a folder)
        (
            stage.folder.relative_to(experiment),
            f"{stage.step}" + ("  (shared)" if stage.shared else ""),
            True,
        )
        for stage in layout.stages
    ]
    entries += [
        (run / CONFIG_FILE, "what this run was made with", False),
        (run / "README.md", "this file", False),
        (run / "logs", "worker logs", True),
    ]
    if pre.save_visualisations:
        entries.append(
            (
                layout.sampling_overviews.relative_to(experiment),
                "sampling thumbnails",
                True,
            )
        )

    tree: dict = {}
    for path, note, is_folder in entries:
        node = tree
        for part in path.parts:
            node = node.setdefault(part, {})
        node[_NOTE] = note
        node[_FOLDER] = is_folder
    rows = _rows(tree, "")
    width = max(len(label) for label, _ in rows)
    return [f"{experiment.name}/"] + [
        f"{label:<{width}}    {note}".rstrip() for label, note in rows
    ]


_NOTE, _FOLDER = "\0note", "\0folder"


def _rows(tree: dict, prefix: str) -> list[tuple[str, str]]:
    names = sorted(name for name in tree if name not in (_NOTE, _FOLDER))
    rows = []
    for i, name in enumerate(names):
        last = i == len(names) - 1
        child = tree[name]
        has_children = any(key not in (_NOTE, _FOLDER) for key in child)
        slash = "/" if has_children or child.get(_FOLDER) else ""
        rows.append(
            (f"{prefix}{'`-- ' if last else '|-- '}{name}{slash}", child.get(_NOTE, ""))
        )
        rows += _rows(child, prefix + ("    " if last else "|   "))
    return rows
