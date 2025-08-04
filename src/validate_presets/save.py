import json
from pathlib import Path

from src.types.preset import Preset
from .test import Result


def append_result(result: Result, preset: Preset, file_path: Path):
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=4)

    with file_path.open("r+", encoding="utf-8") as f:
        data: list = json.load(f)

        r = {
            "preset": str(preset),
            "best_loss": result.best_loss,
            "best_accuracy": result.best_accuracy,
            "mean_loss": result.mean_loss,
            "mean_accuracy": result.mean_accuracy,
        }
        data.append(r)

        f.seek(0)
        json.dump(data, f, indent=4)
