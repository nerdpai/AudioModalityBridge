import sys

from torch.utils.data import DataLoader

from src.types.preset import PresetsTypes
from src.types.dataset import Splits
from src.constants.train import (
    PRESETS_FACTORY,
    #
    BATCH_SIZE,
    NUM_WORKERS,
    #
    NUM_EPOCHS,
    MAX_STEPS,
    MAX_NEW_TOKENS,
    #
    BRIDGE_LEARNING_RATE,
    AUDIO_LEARNING_RATE,
    LEARNING_RATE_FACTOR,
    PATIENCE,
    #
    RESULTS_PATH,
    MODELS_PATH,
)
from src.utils.dataloader import get_loader
from .train import train
from .save import save_model, save_results


def main():
    data_loaders: dict[Splits, DataLoader] = {
        "test": get_loader("test", BATCH_SIZE, NUM_WORKERS),
        "dev": get_loader("dev", BATCH_SIZE, NUM_WORKERS),
        "train": get_loader("train", BATCH_SIZE, NUM_WORKERS),
    }

    model_name: PresetsTypes = sys.argv[1]  # type: ignore

    result_path = RESULTS_PATH / model_name
    model_path = MODELS_PATH / model_name

    factory = PRESETS_FACTORY[model_name]

    result, model = train(
        data_loaders,
        factory,
        NUM_EPOCHS,
        MAX_STEPS,
        MAX_NEW_TOKENS,
        BRIDGE_LEARNING_RATE,
        AUDIO_LEARNING_RATE,
        LEARNING_RATE_FACTOR,
        PATIENCE,
    )

    save_results(result, result_path)
    save_model(model, model_path)


if __name__ == "__main__":
    main()
