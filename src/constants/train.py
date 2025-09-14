from typing import Final, Optional
from pathlib import Path

from src.types.preset import Preset, PresetsTypes
from src.models.presets import create_factory, VoiceLMGen

from src.models.llm.llama import LLama3Model
from src.models.asr.whisper import WhisperModel
from src.models.classification.wav2vec2 import Wav2Vec2Model


# Preset values
PRESETS: dict[PresetsTypes, Preset] = {
    "classification/wav2vec2": Preset(
        num_atten_heads=16,
        translate_chunk_seconds=1.0,
        in_out_rel=8.0,
        overlap_audio_chunks=False,
    ),
    "asr/whisper": Preset(
        num_atten_heads=6,
        translate_chunk_seconds=2.0,
        in_out_rel=8.0,
        overlap_audio_chunks=True,
    ),
}

PRESETS_FACTORY: dict[PresetsTypes, VoiceLMGen] = {
    "classification/wav2vec2": create_factory(
        PRESETS["classification/wav2vec2"],
        Wav2Vec2Model,
        LLama3Model,
    ),
    "asr/whisper": create_factory(
        PRESETS["asr/whisper"],
        WhisperModel,
        LLama3Model,
    ),  # type: ignore
}


# Train constants
MAX_STEPS: Final[Optional[int]] = 500
BATCH_SIZE: Final[int] = 18
NUM_WORKERS: Final[int] = 16
NUM_EPOCHS: Final[int] = 1
MAX_NEW_TOKENS: Final[int] = 128
BRIDGE_LEARNING_RATE: Final[float] = 1e-3
AUDIO_LEARNING_RATE: Final[float] = 5e-4
LEARNING_RATE_FACTOR: Final[float] = 0.9
PATIENCE: Final[int] = 25
RESULTS_PATH: Final[Path] = Path("./results/train").resolve()
MODELS_PATH: Final[Path] = Path("./.models").resolve()
