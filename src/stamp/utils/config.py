from pydantic import BaseModel, ConfigDict

from stamp.encoding.config import PatientEncodingConfig, SlideEncodingConfig
from stamp.modeling.config import (
    AdvancedConfig,
    CrossvalConfig,
    DeploymentConfig,
    TrainConfig,
)
from stamp.preprocessing.config import PreprocessingConfig


class StampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing: PreprocessingConfig | None = None

    slide_encoding: SlideEncodingConfig | None = None

    patient_encoding: PatientEncodingConfig | None = None

    advanced_config: AdvancedConfig | None = None
