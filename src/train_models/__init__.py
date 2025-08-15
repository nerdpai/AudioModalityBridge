import shutil
import sys
import subprocess

from src.constants.train import (
    RESULTS_PATH,
    MODELS_PATH,
    PRESETS_FACTORY,
)
from src.utils.print import title_print
from src.utils.bool import parse_bool


def sanity_check() -> bool:
    if not RESULTS_PATH.exists() and not MODELS_PATH.exists():
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        return True

    should = input(
        f"Result for training: {RESULTS_PATH} or saved models path: {MODELS_PATH} already exists."
        "Should I remove it all? (y/n): "
    )
    if not parse_bool(should):
        print("Exiting without changes.")
        return False

    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    return True


def run():
    if not sanity_check():
        return

    for model_name, presets in PRESETS_FACTORY.items():
        title_print(f"\n\nTrain Model: {model_name}\n")

        cmd = [
            sys.executable,
            "-m",
            "src.train_models",
            model_name,
        ]
        subprocess.run(cmd, check=True)
