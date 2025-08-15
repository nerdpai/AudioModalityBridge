from typing import Final
from pathlib import Path

from src.types.preset import Preset
from src.types.dataset import Splits
from src.utils.collections import named_product


# Preset values
ATTENTION_HEADS = [6, 8, 12, 16]
CHUNK_SECONDS = [5.0, 2.0, 1.0]
IN_OUT_REL = [1.0, 1.5, 2.0, 4.0, 8.0, 16.0]
OVERLAP_AUDIO_CHUNKS = [True, False]

PRESETS: list[Preset] = [
    Preset.from_dict(preset)
    for preset in named_product(
        [ATTENTION_HEADS, CHUNK_SECONDS, IN_OUT_REL, OVERLAP_AUDIO_CHUNKS],
        Preset.fields(),
    )
]

# Validation presets
MAX_STEPS: Final[int] = 50
MAX_GEN_LENGTH: Final[int] = 128
BATCH_SIZE: Final[int] = 6
DATA_SPLIT: Final[Splits] = "train"
NUM_WORKERS: Final[int] = 20
LEARNING_RATE: Final[float] = 1e-3
RESULTS_PATH: Final[Path] = Path("./results/presets").resolve()
