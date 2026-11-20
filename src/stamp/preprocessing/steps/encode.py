"""Step `encode`: patient bag with features -> one embedding per patient (e.g. TITAN).

Equivalent to `stamp encode_slides` on the patient bags (each patient is one
virtual slide), except that padding rows and repeated tiles are left out, and
the patient's label and IDs are stored with the embedding.

Encoder memory grows quickly with the number of tiles, so unsampled bags of
large patients may not fit on the GPU. Either sample (bags are bounded by
n_tiles) or set a chunk size: the tiles are then encoded in random chunks and
the chunk embeddings averaged, which gives the same fixed-size embedding.
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
from stamp.preprocessing.steps.sample import patient_rng
from stamp.types import Microns, TilePixels


@dataclass(frozen=True)
class EncodingSettings:
    device: str
    chunk_size: int | None = None
    """See `EncodingConfig.chunk_size`."""

    seed: int = 0


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
    _write_embedding(output, embedding, bag, model, settings)


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

    embeddings = [
        _encode_rows(bag, rows, encoder, settings.device)
        for rows in chunks_of(bag, settings)
    ]
    if len(embeddings) == 1:
        return embeddings[0]
    mean = np.mean(np.stack(embeddings).astype(np.float64), axis=0)
    return mean.astype(embeddings[0].dtype)


def chunks_of(bag: Bag, settings: EncodingSettings) -> list[npt.NDArray[np.int64]]:
    """Rows of the bag's distinct tiles for each encoder call, in bag order.

    One chunk with every tile, unless `chunk_size` is set and the bag is larger:
    then random chunks of (nearly) equal size, none above `chunk_size`.
    """
    rows, _ = bag.distinct_tiles()
    if settings.chunk_size is None or len(rows) <= settings.chunk_size:
        return [rows]
    group_key = bag.patient.group_key if bag.patient is not None else ""
    shuffled = patient_rng(settings.seed, group_key).permutation(rows)
    n_chunks = -(-len(rows) // settings.chunk_size)
    return [np.sort(chunk) for chunk in np.array_split(shuffled, n_chunks)]


def _encode_rows(
    bag: Bag, rows: npt.NDArray[np.int64], encoder: Encoder, device: str
) -> npt.NDArray[np.floating]:
    assert bag.feats is not None
    feats = torch.from_numpy(bag.feats[rows]).to(dtype=encoder.precision)
    coords = CoordsInfo(
        bag.coords[rows], Microns(bag.tile_size_um), TilePixels(bag.tile_size_px)
    )
    # Same call `stamp encode_slides` makes for each slide.
    return encoder._generate_slide_embedding(feats, device, coords=coords)


def _write_embedding(
    output: Path,
    embedding: npt.NDArray[np.floating],
    bag: Bag,
    encoder: Encoder,
    settings: EncodingSettings,
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
                "n_chunks": len(chunks_of(bag, settings)),
                "stamp_version": stamp.__version__,
            }
        )
        if bag.patient is not None:
            f.attrs.update(patient_attrs(bag.patient))
        if bag.sampling is not None:
            f.attrs["sampling"] = json.dumps(dataclasses.asdict(bag.sampling))
