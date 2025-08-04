import torch
from transformers.models.wav2vec2 import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)

from src.models.audio_model import AudioModel, AudioModelConfig
from src.constants.models import MODELS


class Wav2Vec2Model(AudioModel):
    def __init__(self, device: torch.device = torch.device("cpu")):
        config = self.get_config()
        card = config.model_card.asdict()

        model = Wav2Vec2ForCTC.from_pretrained(
            **card,
            device_map=device,
        )

        encoder = model.wav2vec2
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(**card)

        super().__init__(
            encoder,
            feature_extractor,
        )

    @staticmethod
    def get_config() -> AudioModelConfig:
        return MODELS.asr.wav2vec2
