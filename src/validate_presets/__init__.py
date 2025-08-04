import shutil

from src.constants.presets import (
    BATCH_SIZE,
    DATA_SPLIT,
    NUM_WORKERS,
    LEARNING_RATE,
    RESULTS_PATH,
)
from src.utils.print import title_print
from src.utils.bool import parse_bool
from src.models.presets import PRESETS_FACTORY
from .dataloader import get_loader
from .test import test_preset
from .save import append_result


def sanity_check() -> bool:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        should = input(
            f"Result for presets: {RESULTS_PATH} already exists. Should I remove it? (y/n): "
        )
        if not parse_bool(should):
            print("Exiting without changes.")
            return False

        shutil.rmtree(RESULTS_PATH)
        return True
    return True


def run():
    title_print("Prepare Dataset\n")
    data_loader = get_loader(DATA_SPLIT, BATCH_SIZE, NUM_WORKERS)
    print("Done.")

    if not sanity_check():
        return

    for model_name, presets in PRESETS_FACTORY.items():
        title_print(f"\n\nRun Preset: {model_name}\n")
        save_file = RESULTS_PATH / f"{model_name}.json"

        for factory in presets:
            result = test_preset(data_loader, factory[1], LEARNING_RATE)
            append_result(result, factory[0], save_file)
