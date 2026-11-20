"""Small stand-ins for the feature extractor and slide encoder (no weights needed)."""

import numpy as np
import torch
from PIL import Image
from torch import nn

from stamp.preprocessing.extractor import Extractor

FEATURE_DIM = 6


def to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(pixels).permute(2, 0, 1)


class MeanStd(nn.Module):
    """A 6-number 'feature': per-channel mean and standard deviation of the tile."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)


def tile_feature(image: Image.Image) -> np.ndarray:
    return MeanStd()(to_tensor(image)[None]).numpy()[0]


def fake_extractor() -> Extractor:
    return Extractor(model=MeanStd(), transform=to_tensor, identifier="fake")


class FakeEncoder:
    """Embedding = mean tile feature. Records the inputs of every call."""

    identifier = "fake_encoder"
    precision = torch.float32
    required_extractors = ["fake"]

    def __init__(self) -> None:
        self.model = nn.Identity()
        self.calls: list[tuple[torch.Tensor, np.ndarray]] = []

    def _generate_slide_embedding(self, feats, device, coords=None, **kwargs):
        self.calls.append((feats, coords.coords_um.copy()))
        return feats.mean(0).numpy()
