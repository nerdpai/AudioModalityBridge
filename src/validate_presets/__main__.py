import sys

from src.types.preset import PresetsTypes
from src.constants.presets import (
    MAX_STEPS,
    MAX_GEN_LENGTH,
    BATCH_SIZE,
    DATA_SPLIT,
    NUM_WORKERS,
    LEARNING_RATE,
    RESULTS_PATH,
)
from src.utils.dataloader import get_loader
from src.models.presets import PRESETS_FACTORY
from .test import test_preset
from .save import append_result


def main():
    data_loader = get_loader(DATA_SPLIT, BATCH_SIZE, NUM_WORKERS)

    model_name: PresetsTypes = sys.argv[1]  # type: ignore
    preset_id = int(sys.argv[2])

    save_file = RESULTS_PATH / f"{model_name}.json"
    factory = PRESETS_FACTORY[model_name][preset_id]

    result = test_preset(
        data_loader,
        factory[1],
        MAX_STEPS,
        MAX_GEN_LENGTH,
        LEARNING_RATE,
    )
    append_result(result, factory[0], save_file)


if __name__ == "__main__":
    main()
