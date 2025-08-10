from typing import Final
from pathlib import Path

from src.types.preset import Preset, PresetsTypes
from src.models.presets import create_factory, VoiceLMGen

from src.models.llm.llama import LLama3Model
from src.models.classification.ast import ASTModel
from src.models.asr.whisper import WhisperModel


# Preset values
PRESETS: dict[PresetsTypes, Preset] = {
    "classification/ast": Preset(
        num_atten_heads=8,
        translate_chunk_seconds=5.0,
        in_out_rel=8.0,
        overlap_audio_chunks=True,
    ),
    "asr/whisper": Preset(
        num_atten_heads=6,
        translate_chunk_seconds=2.0,
        in_out_rel=8.0,
        overlap_audio_chunks=True,
    ),
}

PRESETS_FACTORY: dict[PresetsTypes, VoiceLMGen] = {
    "classification/ast": create_factory(
        PRESETS["classification/ast"],
        ASTModel,
        LLama3Model,
    ),
    "asr/whisper": create_factory(
        PRESETS["asr/whisper"],
        WhisperModel,
        LLama3Model,
    ),
}  # type: ignore


# Validation presets
BATCH_SIZE: Final[int] = 12
NUM_WORKERS: Final[int] = 8
NUM_EPOCHS: Final[int] = 1
BRIDGE_LEARNING_RATE: Final[float] = 1e-3
AUDIO_LEARNING_RATE: Final[float] = 5e-4
LEARNING_RATE_FACTOR: Final[float] = 0.9
PATIENCE: Final[int] = 10
RESULTS_PATH: Final[Path] = Path("./results/train").resolve()
MODELS_PATH: Final[Path] = Path("./.models").resolve()
