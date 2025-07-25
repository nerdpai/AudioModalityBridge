from typing import Final
from pathlib import Path

from .types import Splits

# FETCH constants
REPO_HASH: Final[str] = "3a1ecdc235db65ef38a7e97dd04b603ea68a5810"
BASE_URL: Final[Path] = Path(
    f"huggingface.co/datasets/mozilla-foundation/common_voice_11_0/resolve/{REPO_HASH}"
)
LOCALE: Final[str] = "en"
NUMBER_OF_SHARDS: dict[Splits, int] = {
    "train": 23,
    "dev": 1,
    "test": 1,
}

# PROCESS constants
UP_VOTES: Final[int] = 5
DOWN_VOTES: Final[int] = 0

# PATHES constants
CACHE_DIR: Final[Path] = Path("./.cache/mozilla_common_voice").resolve()
DATASET_DIR: Final[Path] = Path("./.datasets/mozilla_common_voice").resolve()
