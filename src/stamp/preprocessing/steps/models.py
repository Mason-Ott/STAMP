"""Loading feature extractors and slide encoders, only when first needed.

A machine that finds nothing left to do never loads a model, which matters when
many machines are started on the same run.
"""

import importlib
import logging
from collections.abc import Callable
from typing import Generic, TypeVar

import torch

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor

_logger = logging.getLogger("stamp")

T = TypeVar("T")

_ENCODER_CLASSES: dict[EncoderName, tuple[str, str]] = {
    EncoderName.TITAN: ("titan", "Titan"),
    EncoderName.GIGAPATH: ("gigapath", "Gigapath"),
    EncoderName.CHIEF_CTRANSPATH: ("chief", "CHIEF"),
    EncoderName.MADELEINE: ("madeleine", "Madeleine"),
    EncoderName.PRISM: ("prism", "Prism"),
}
"""Encoder -> (module in stamp.encoding.encoder, class). EAGLE is left out: it
needs a second set of features that the pipeline doesn't produce."""


ENCODER_EXTRACTORS: dict[EncoderName, frozenset[ExtractorName]] = {
    EncoderName.TITAN: frozenset({ExtractorName.CONCH1_5}),
    EncoderName.GIGAPATH: frozenset({ExtractorName.GIGAPATH}),
    EncoderName.CHIEF_CTRANSPATH: frozenset({ExtractorName.CHIEF_CTRANSPATH}),
    EncoderName.MADELEINE: frozenset({ExtractorName.CONCH}),
    EncoderName.PRISM: frozenset({ExtractorName.VIRCHOW_FULL}),
}
"""Features each encoder accepts (its `required_extractors`), known without
loading the encoder, so incompatible runs can skip encoding up front."""


def encoder_accepts(encoder: EncoderName, extractor: ExtractorName) -> bool:
    return extractor in ENCODER_EXTRACTORS.get(encoder, frozenset({extractor}))


class Lazy(Generic[T]):
    """Calls `factory` on the first `get()` and returns the same object afterwards."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._value: T | None = None

    def get(self) -> T:
        if self._value is None:
            self._value = self._factory()
        return self._value


def load_extractor(name: ExtractorName, device: str) -> Extractor:
    """The same extractor `stamp preprocess` would use, moved to `device`.

    Relies on STAMP's convention that extractor `x-y` is created by the function
    `x_y()` in `stamp.preprocessing.extractor.x_y`.
    """
    factory_name = str(name).replace("-", "_")
    module = importlib.import_module(f"stamp.preprocessing.extractor.{factory_name}")
    _logger.info(f"loading extractor {name} on {device}")
    extractor: Extractor = getattr(module, factory_name)()
    extractor.model.to(device).eval()
    return extractor


def load_encoder(name: EncoderName, device: str) -> Encoder:
    if name not in _ENCODER_CLASSES:
        raise ValueError(f"encoder {name} is not supported by the pipeline")
    module_name, class_name = _ENCODER_CLASSES[name]
    module = importlib.import_module(f"stamp.encoding.encoder.{module_name}")
    _logger.info(f"loading encoder {name} on {device}")
    encoder: Encoder = getattr(module, class_name)()
    encoder.model.to(device).eval()
    if encoder.precision == torch.float16:
        encoder.model.half()
    return encoder
