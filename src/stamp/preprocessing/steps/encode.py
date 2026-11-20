"""Step `encode`: patient bag with features -> one embedding per patient (e.g. TITAN).

Equivalent to `stamp encode_slides` on the patient bags (each patient is one
virtual slide), except that padding rows and repeated tiles are left out, and
the patient's label and IDs are stored with the embedding.

Encoder memory grows quickly with the number of tiles, so unsampled bags of
large patients may not fit on the GPU; sampled bags are bounded by n_tiles.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from stamp.encoding.encoder import Encoder, _resolve_extractor_name
from stamp.modeling.data import CoordsInfo
from stamp.preprocessing.data.bags import Bag, patient_attrs, read_bag
from stamp.preprocessing.data.files import atomic_h5_file
from stamp.preprocessing.steps.models import Lazy
from stamp.types import Microns, TilePixels


@dataclass(frozen=True)
class EncodingSettings:
    device: str


def run(
    item: object,
    inputs: list[Path],
    output: Path,
    *,
    encoder: Lazy[Encoder],
    settings: EncodingSettings,
) -> None:
    """Step entry point. `inputs` holds the patient's bag with features."""
    (bag_file,) = inputs
    bag = read_bag(bag_file)
    model = encoder.get()
    embedding = encode_bag(bag, model, settings)
    _write_embedding(output, embedding, bag, model)


def encode_bag(
    bag: Bag, encoder: Encoder, settings: EncodingSettings
) -> npt.NDArray[np.floating]:
    if bag.feats is None:
        raise ValueError("bag has no features to encode")
    extractor = _resolve_extractor_name(bag.extractor or "")
    if extractor not in encoder.required_extractors:
        raise ValueError(
            f"{encoder.identifier} needs features from {encoder.required_extractors}, "
            f"not {bag.extractor}"
        )

    rows, _ = bag.distinct_tiles()
    feats = torch.from_numpy(bag.feats[rows]).to(dtype=encoder.precision)
    coords = CoordsInfo(
        bag.coords[rows], Microns(bag.tile_size_um), TilePixels(bag.tile_size_px)
    )
    # Same call `stamp encode_slides` makes for each slide.
    return encoder._generate_slide_embedding(feats, settings.device, coords=coords)


def _write_embedding(
    output: Path, embedding: npt.NDArray[np.floating], bag: Bag, encoder: Encoder
) -> None:
    import stamp

    with atomic_h5_file(output) as f:
        f["feats"] = embedding
        f.attrs.update(
            {
                "feat_type": "patient",
                "encoder": str(encoder.identifier),
                "extractor": str(bag.extractor),
                "precision": str(encoder.precision),
                "n_tiles": len(bag.distinct_tiles()[0]),
                "stamp_version": stamp.__version__,
            }
        )
        if bag.patient is not None:
            f.attrs.update(patient_attrs(bag.patient))
        if bag.sampling is not None:
            f.attrs["sampling"] = json.dumps(dataclasses.asdict(bag.sampling))
