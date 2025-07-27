from typing import Final
from pathlib import Path

from ..types.dataset import Splits, Limits

# FETCH constants
REPO_HASH: Final[str] = "3a1ecdc235db65ef38a7e97dd04b603ea68a5810"
BASE_URL: Final[Path] = Path(
    f"huggingface.co/datasets/mozilla-foundation/common_voice_11_0/resolve/{REPO_HASH}"
)
LOCALE: Final[str] = "en"

LIMITS: dict[Splits, Limits] = {
    "train": Limits(
        UP_VOTES=4,
        DOWN_VOTES=0,
        NUMBER_OF_SHARDS=0,
    ),
    "dev": Limits(
        UP_VOTES=4,
        DOWN_VOTES=0,
        NUMBER_OF_SHARDS=0,
    ),
    "test": Limits(
        UP_VOTES=4,
        DOWN_VOTES=0,
        NUMBER_OF_SHARDS=1,
    ),
}

# PATHES constants
CACHE_DIR: Final[Path] = Path("./.cache/mozilla_common_voice").resolve()
DATASET_DIR: Final[Path] = Path("./.datasets/mozilla_common_voice").resolve()
