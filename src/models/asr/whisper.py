import torch
from transformers.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)

from src.models.audio_model import AudioModel, AudioModelConfig
from src.constants.models import MODELS


class WhisperModel(AudioModel):
    def __init__(self, device: torch.device = torch.device("cpu")):
        config = self.get_config()
        card = config.model_card.asdict()

        model = WhisperForConditionalGeneration.from_pretrained(
            **card,
            device_map=device,
        )

        encoder = model.model.encoder
        feature_extractor = WhisperFeatureExtractor.from_pretrained(**card)

        super().__init__(
            encoder,
            feature_extractor,
        )

    @staticmethod
    def get_config() -> AudioModelConfig:
        return MODELS.asr.whisper
