from typing import Protocol, TYPE_CHECKING, Optional
from functools import wraps

from transformers.modeling_utils import PreTrainedModel
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
import torch
from torch import nn

from src.types.models import AudioModelConfig


class AudioModel(nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        feature_extractor: SequenceFeatureExtractor,
    ):
        super().__init__()

        self.encoder = encoder
        self.feature_extractor = feature_extractor

    @property
    def config(self) -> AudioModelConfig:
        return self.get_config()

    @staticmethod
    def get_config() -> AudioModelConfig:
        raise NotImplementedError("Child did not implement the method")

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):  # type: ignore
        self.encoder.to(*args, **kwargs)
        return self

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        kwargs = (
            {"attention_mask": attention_mask} if attention_mask is not None else {}
        )
        return self.encoder(
            inputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


if TYPE_CHECKING:

    class AudioModelProtocol(AudioModel, Protocol):  # type: ignore
        def __init__(self, device: torch.device = torch.device("cpu")) -> None: ...

else:

    class AudioModelProtocol(Protocol):
        def __init__(self, device: torch.device = torch.device("cpu")) -> None: ...
