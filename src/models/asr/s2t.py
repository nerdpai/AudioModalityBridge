import torch
from transformers.models.speech_to_text import (
    Speech2TextFeatureExtractor,
    Speech2TextForConditionalGeneration,
)

from src.models.audio_model import AudioModel, AudioModelConfig
from src.constants.models import MODELS


class S2TModel(AudioModel):
    def __init__(self, device: torch.device = torch.device("cpu")):
        config = self.get_config()
        card = config.model_card.asdict()

        model = Speech2TextForConditionalGeneration.from_pretrained(
            **card,
            device_map=device,
        )

        encoder = model.model.encoder
        feature_extractor = Speech2TextFeatureExtractor.from_pretrained(**card)

        super().__init__(
            encoder,
            feature_extractor,
        )

    @staticmethod
    def get_config() -> AudioModelConfig:
        return MODELS.asr.s2t
