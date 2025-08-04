import torch
from transformers.models.audio_spectrogram_transformer import (
    ASTForAudioClassification,
    ASTFeatureExtractor,
)
from src.models.audio_model import AudioModel, AudioModelConfig
from src.constants.models import MODELS


class ASTModel(AudioModel):
    def __init__(self, device: torch.device = torch.device("cpu")):
        config = self.get_config()
        card = config.model_card.asdict()

        model = ASTForAudioClassification.from_pretrained(
            **card,
            device_map=device,
        )

        encoder = model.audio_spectrogram_transformer
        feature_extractor = ASTFeatureExtractor.from_pretrained(**card)

        super().__init__(
            encoder,
            feature_extractor,
        )

    @staticmethod
    def get_config() -> AudioModelConfig:
        return MODELS.classification.ast
