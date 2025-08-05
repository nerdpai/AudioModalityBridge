import shutil
import sys
import subprocess

from src.constants.presets import (
    RESULTS_PATH,
)
from src.utils.print import title_print
from src.utils.bool import parse_bool
from src.models.presets import PRESETS_FACTORY


def sanity_check() -> bool:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        return True

    should = input(
        f"Result for presets: {RESULTS_PATH} already exists. Should I remove it? (y/n): "
    )
    if not parse_bool(should):
        print("Exiting without changes.")
        return False

    shutil.rmtree(RESULTS_PATH)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    return True


def run():
    if not sanity_check():
        return

    for model_name, presets in PRESETS_FACTORY.items():
        title_print(f"\n\nRun Preset: {model_name}\n")

        for preset_id in range(len(presets)):
            cmd = [
                sys.executable,
                "-m",
                "src.validate_presets",
                model_name,
                str(preset_id),
            ]
            subprocess.run(cmd, check=True)
