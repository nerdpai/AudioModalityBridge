from typing import Literal, Optional
from dataclasses import dataclass, asdict

import torch


@dataclass(frozen=True)
class ModelCard:
    pretrained_model_name_or_path: str
    revision: str
    token: str
    torch_dtype: torch.dtype = torch.bfloat16

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class AudioModelConfig:
    model_card: ModelCard
    embed_dim: int
    feature_property_name: Literal["input_features", "input_values"]
    seq_per_second: int


@dataclass(frozen=True)
class LanguageModelConfig:
    model_card: ModelCard
    embed_dim: int
