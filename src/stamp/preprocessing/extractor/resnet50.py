import torch
import torchvision
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from jaxtyping import Float

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor


class _ResNet50(torch.nn.Module):
    """ResNet50 with the classification head removed, outputting 2048-dim features."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(
        self, batch: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch feature"]:
        x = self.base.conv1(batch)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        return torch.flatten(x, 1)  # (batch, 2048)


def resnet50() -> Extractor[_ResNet50]:
    base = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return Extractor(
        model=_ResNet50(base),
        transform=transform,
        identifier=ExtractorName.RESNET50,
    )