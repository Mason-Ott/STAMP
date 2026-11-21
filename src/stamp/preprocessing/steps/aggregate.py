"""Step `aggregate`: a patient's slide embeddings -> one embedding per patient.

Reads the file the `encode` step wrote with per-slide encoding (`slide_feats`,
one row per slide) and combines the rows:

    mean    average of the slides' embeddings          (D,)
    max     element-wise maximum                       (D,)
    concat  slide embeddings one after the other,      (max_slides * D,)
            zero-padded for patients with fewer slides

The output is a patient embedding file like the one `encode` writes without
per-slide encoding, so downstream code reads it the same way.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from stamp.preprocessing.data.files import atomic_h5_file
from stamp.preprocessing.pipeline.config import SlideAggregation


@dataclass(frozen=True)
class AggregationSettings:
    method: SlideAggregation
    max_slides: int | None = None
    """concat only: number of slide slots."""


def run(
    item: object,
    inputs: list[Path],
    output: Path,
    *,
    settings: AggregationSettings,
) -> None:
    """Step entry point. `inputs` holds the patient's per-slide embeddings."""
    (slide_file,) = inputs
    with h5py.File(slide_file) as source:
        slide_feats = source["slide_feats"][:]
        attrs = dict(source.attrs)
    slide_ids = json.loads(attrs["slide_ids"])
    embedding = aggregate_embeddings(
        slide_feats, settings, patient=attrs.get("group_key")
    )

    attrs.update(
        {
            "feat_type": "patient",
            "slide_aggregation": settings.method,
            "n_slides": len(slide_ids),
        }
    )
    with atomic_h5_file(output) as f:
        f["feats"] = embedding
        f.attrs.update(attrs)


def aggregate_embeddings(
    slide_feats: npt.NDArray[np.floating],
    settings: AggregationSettings,
    *,
    patient: object = None,
) -> npt.NDArray[np.floating]:
    """(n_slides, D) -> (D,), or (max_slides * D,) for concat."""
    match settings.method:
        case "mean":
            return slide_feats.astype(np.float64).mean(axis=0).astype(slide_feats.dtype)
        case "max":
            return slide_feats.max(axis=0)
        case "concat":
            assert settings.max_slides is not None
            n_slides, dim = slide_feats.shape
            if n_slides > settings.max_slides:
                raise ValueError(
                    f"patient {patient} has {n_slides} encoded slides, more than "
                    f"encoding.max_slides = {settings.max_slides}"
                )
            padded = np.zeros((settings.max_slides, dim), slide_feats.dtype)
            padded[:n_slides] = slide_feats
            return padded.reshape(-1)
