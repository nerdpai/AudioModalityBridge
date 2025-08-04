from dataclasses import dataclass

from src.types.models import ModelCard, AudioModelConfig, LanguageModelConfig
from .secrets import SECRETS

TOKEN = SECRETS["HF_ACCESS_TOKEN"]


@dataclass(frozen=True)
class ClassificationModels:
    wav2vec2 = AudioModelConfig(
        ModelCard(
            "somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd",
            revision="9bc54869d8082fc35adcbbf6f603e8365621f8e0",
            token=TOKEN,
        ),
        embed_dim=768,
        feature_property_name="input_values",
        seq_per_second=50,
    )
    ast = AudioModelConfig(
        ModelCard(
            "aicinema69/audio-emotion-detector-try2",
            revision="1913c7d7957c8f5c1e32f20f168a3d12ec9530fb",
            token=TOKEN,
        ),
        embed_dim=768,
        feature_property_name="input_values",
        seq_per_second=22,
    )


@dataclass(frozen=True)
class ASRModels:
    wav2vec2 = AudioModelConfig(
        ModelCard(
            "facebook/wav2vec2-large-960h",
            revision="bdeaacdf88f7a155f50a2704bc967aa81fbbb2ab",
            token=TOKEN,
        ),
        embed_dim=1024,
        feature_property_name="input_values",
        seq_per_second=50,
    )
    s2t = AudioModelConfig(
        ModelCard(
            "facebook/s2t-large-librispeech-asr",
            revision="a4b4750ad1425acda0dbd1daa9188d5fd7872491",
            token=TOKEN,
        ),
        embed_dim=1024,
        feature_property_name="input_features",
        seq_per_second=25,
    )
    whisper = AudioModelConfig(
        ModelCard(
            "openai/whisper-small.en",
            revision="e8727524f962ee844a7319d92be39ac1bd25655a",
            token=TOKEN,
        ),
        embed_dim=768,
        feature_property_name="input_features",
        seq_per_second=50,
    )


@dataclass(frozen=True)
class LLMModels:
    llama3 = LanguageModelConfig(
        ModelCard(
            "meta-llama/Llama-3.2-1B-Instruct",
            revision="9213176726f574b556790deb65791e0c5aa438b6",
            token=TOKEN,
        ),
        embed_dim=2048,
    )


@dataclass(frozen=True)
class Models:
    classification = ClassificationModels()
    asr = ASRModels()
    llm = LLMModels()


MODELS = Models()
