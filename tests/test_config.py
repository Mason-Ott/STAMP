# %%
from pathlib import Path

from stamp.modeling.config import (
    AdvancedConfig,
    CrossvalConfig,
    DeploymentConfig,
    MlpModelParams,
    ModelParams,
    TrainConfig,
    TransMILModelParams,
    VitModelParams,
)
from stamp.preprocessing.config import (
    ExtractorName,
    Microns,
    PreprocessingConfig,
    SlideMPP,
    TilePixels,
)
from stamp.utils.config import StampConfig


def test_config_parsing() -> None:
    config = StampConfig.model_validate(
        {
            "preprocessing": {
                "brightness_cutoff": 240,
                "cache_dir": "cache",
                "canny_cutoff": 0.02,
                "device": "cuda",
                "extractor": "ctranspath",
                "max_workers": 8,
                "output_dir": "bla",
                "tile_size_px": 224,
                "tile_size_um": 256.0,
                "wsi_dir": "wsis",
                "default_slide_mpp": 1.0,
            },
            "advanced_config": {
                "seed": 42,
                "bag_size": 512,
                "num_workers": 16,
                "batch_size": 64,
                "max_epochs": 64,
                "patience": 16,
                "accelerator": "gpu",
                "model_params": {
                    "vit": {
                        "dim_model": 512,
                        "dim_feedforward": 512,
                        "n_heads": 8,
                        "n_layers": 2,
                        "dropout": 0.25,
                        "use_alibi": True,
                    },
                    "mlp": {
                        "dim_hidden": 512,
                        "num_layers": 2,
                        "dropout": 0.25,
                    },
                },
            },
        }
    )

    assert config == StampConfig(
        preprocessing=PreprocessingConfig(
            output_dir=Path("bla"),
            wsi_dir=Path("wsis"),
            cache_dir=Path("cache"),
            tile_size_um=Microns(256.0),
            tile_size_px=TilePixels(224),
            extractor=ExtractorName.CTRANSPATH,
            max_workers=8,
            device="cuda",
            brightness_cutoff=240,
            canny_cutoff=0.02,
            default_slide_mpp=SlideMPP(1.0),
        ),
        advanced_config=AdvancedConfig(
            seed=42,
            bag_size=512,
            num_workers=16,
            batch_size=64,
            max_epochs=64,
            patience=16,
            accelerator="gpu",
            model_params=ModelParams(
                vit=VitModelParams(
                    dim_model=512,
                    dim_feedforward=512,
                    n_heads=8,
                    n_layers=2,
                    dropout=0.25,
                    use_alibi=True,
                ),
                mlp=MlpModelParams(
                    dim_hidden=512,
                    num_layers=2,
                    dropout=0.25,
                ),
                trans_mil=TransMILModelParams(dim_hidden=512),
            ),
        ),
    )
