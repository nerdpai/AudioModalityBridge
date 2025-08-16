from typing import Final
from pathlib import Path

from src.types.preset import Preset, PresetsTypes
from src.models.presets import create_factory, VoiceLMGen

from src.models.llm.llama import LLama3Model
from src.models.classification.wav2vec2 import Wav2Vec2Model as CLSWav2Vec2Model
from src.models.asr.wav2vec2 import Wav2Vec2Model as ASRWav2Vec2Model


# Preset values
PRESETS: dict[PresetsTypes, Preset] = {
    "classification/wav2vec2": Preset(
        num_atten_heads=16,
        translate_chunk_seconds=5.0,
        in_out_rel=8.0,
        overlap_audio_chunks=False,
    ),
    "asr/wav2vec2": Preset(
        num_atten_heads=8,
        translate_chunk_seconds=2.0,
        in_out_rel=16.0,
        overlap_audio_chunks=True,
    ),
}

PRESETS_FACTORY: dict[PresetsTypes, VoiceLMGen] = {
    "classification/wav2vec2": create_factory(
        PRESETS["classification/wav2vec2"],
        CLSWav2Vec2Model,
        LLama3Model,
    ),
    "asr/wav2vec2": create_factory(
        PRESETS["asr/wav2vec2"],
        ASRWav2Vec2Model,
        LLama3Model,
    ),
}  # type: ignore


# Validation presets
BATCH_SIZE: Final[int] = 18
NUM_WORKERS: Final[int] = 16
NUM_EPOCHS: Final[int] = 1
MAX_NEW_TOKENS: Final[int] = 128
BRIDGE_LEARNING_RATE: Final[float] = 1e-3
AUDIO_LEARNING_RATE: Final[float] = 5e-4
LEARNING_RATE_FACTOR: Final[float] = 0.9
PATIENCE: Final[int] = 20
RESULTS_PATH: Final[Path] = Path("./results/train").resolve()
MODELS_PATH: Final[Path] = Path("./.models").resolve()
