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

    per_slide: bool = False
    """Encode each slide on its own (see `EncodingConfig.slide_aggregation`)."""


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
    """The patient's embedding (D,), or with `per_slide` one row per slide that
    has tiles (S, D), in slide order."""
    if bag.feats is None:
        raise ValueError("bag has no features to encode")
    extractor = _resolve_extractor_name(bag.extractor or "")
    if extractor not in encoder.required_extractors:
        raise ValueError(
            f"{encoder.identifier} needs features from {encoder.required_extractors}, "
            f"not {bag.extractor}"
        )

    embeddings = [
        _encode_group(bag, rows, key, encoder, settings)
        for _, rows, key in _groups(bag, settings)
    ]
    return np.stack(embeddings) if settings.per_slide else embeddings[0]


def encoded_slide_ids(bag: Bag, settings: EncodingSettings) -> list[str]:
    """The slides `encode_bag` returns a row for with `per_slide`."""
    return [bag.slides[k].slide_id for k, _, _ in _groups(bag, settings)]


def chunks_of(bag: Bag, settings: EncodingSettings) -> list[npt.NDArray[np.int64]]:
    """Rows of the bag's distinct tiles for each encoder call, in bag order, for
    the patient as a whole (see `_chunks`)."""
    rows, _ = bag.distinct_tiles()
    return _chunks(rows, settings, _group_key(bag))


def _groups(
    bag: Bag, settings: EncodingSettings
) -> list[tuple[int, npt.NDArray[np.int64], str]]:
    """(slide index, rows, shuffle key) of everything encoded as one unit: the
    whole patient, or each slide that has tiles."""
    rows, _ = bag.distinct_tiles()
    if not settings.per_slide:
        return [(-1, rows, _group_key(bag))]
    groups = [
        (k, rows[bag.slide_idx[rows] == k], f"{_group_key(bag)}/{slide.slide_id}")
        for k, slide in enumerate(bag.slides)
    ]
    if not (groups := [g for g in groups if len(g[1])]):
        raise ValueError("bag has no tiles to encode")
    return groups


def _group_key(bag: Bag) -> str:
    return bag.patient.group_key if bag.patient is not None else ""


def _chunks(
    rows: npt.NDArray[np.int64], settings: EncodingSettings, key: str
) -> list[npt.NDArray[np.int64]]:
    """One chunk with every row, unless `chunk_size` is set and there are more
    rows: then random chunks of (nearly) equal size, none above `chunk_size`."""
    if settings.chunk_size is None or len(rows) <= settings.chunk_size:
        return [rows]
    shuffled = patient_rng(settings.seed, key).permutation(rows)
    n_chunks = -(-len(rows) // settings.chunk_size)
    return [np.sort(chunk) for chunk in np.array_split(shuffled, n_chunks)]


def _encode_group(
    bag: Bag,
    rows: npt.NDArray[np.int64],
    key: str,
    encoder: Encoder,
    settings: EncodingSettings,
) -> npt.NDArray[np.floating]:
    """Encode the rows (in chunks if needed) and average the chunk embeddings."""
    # A slide's tiles are in the slide's own coordinates; a patient's in the
    # virtual layout.
    coords = bag.coords_slide if settings.per_slide else bag.coords
    embeddings = [
        _encode_rows(bag, chunk, coords, encoder, settings.device)
        for chunk in _chunks(rows, settings, key)
    ]
    if len(embeddings) == 1:
        return embeddings[0]
    mean = np.mean(np.stack(embeddings).astype(np.float64), axis=0)
    return mean.astype(embeddings[0].dtype)


def _encode_rows(
    bag: Bag,
    rows: npt.NDArray[np.int64],
    coords_um: npt.NDArray[np.float64],
    encoder: Encoder,
    device: str,
) -> npt.NDArray[np.floating]:
    assert bag.feats is not None
    feats = torch.from_numpy(bag.feats[rows]).to(dtype=encoder.precision)
    coords = CoordsInfo(
        coords_um[rows], Microns(bag.tile_size_um), TilePixels(bag.tile_size_px)
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

    n_chunks = sum(
        len(_chunks(rows, settings, key)) for _, rows, key in _groups(bag, settings)
    )
    with atomic_h5_file(output) as f:
        if settings.per_slide:
            f["slide_feats"] = embedding
            f.attrs["slide_ids"] = json.dumps(encoded_slide_ids(bag, settings))
            f.attrs["feat_type"] = "patient_slides"
        else:
            f["feats"] = embedding
            f.attrs["feat_type"] = "patient"
        f.attrs.update(
            {
                "encoder": str(encoder.identifier),
                "extractor": str(bag.extractor),
                "precision": str(encoder.precision),
                "n_tiles": len(bag.distinct_tiles()[0]),
                "n_chunks": n_chunks,
                "stamp_version": stamp.__version__,
            }
        )
        if bag.patient is not None:
            f.attrs.update(patient_attrs(bag.patient))
        if bag.sampling is not None:
            f.attrs["sampling"] = json.dumps(dataclasses.asdict(bag.sampling))
