from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from src.utils.model_io import save_torch, save_pickle
from src.models.voicelm import VoiceLM
from .train import Results, Result


PlotTypes = Literal["train", "validation"]


def save_model(
    model: VoiceLM,
    model_path: Path,
) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    save_torch(model, model_path / "model.pt")


def plot_data(
    loss: list[float],
    accuracy: list[float],
    xlabel: str,
    result_path: Path,
    plot_type: PlotTypes,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "red"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(loss, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = "blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(accuracy, color=color, label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"{plot_type.capitalize()} Metrics per {xlabel}")
    fig.tight_layout()
    plt.savefig(result_path / f"{plot_type}_{xlabel.lower()}.png")
    plt.close()


def plot_result(result: Result, plot_type: PlotTypes, result_path: Path) -> None:
    result_path.mkdir(parents=True, exist_ok=True)
    loss: list[list[float]] = []
    accuracy: list[list[float]] = []

    if plot_type == "train":
        loss = result.train_loss
        accuracy = result.train_accuracy
    elif plot_type == "validation":
        loss = result.dev_loss
        accuracy = result.dev_accuracy

    loss_epoch: list[float] = [np.mean(epoch).item() for epoch in loss]
    accuracy_epoch: list[float] = [np.mean(epoch).item() for epoch in accuracy]
    loss_batch: list[float] = np.concatenate(loss).tolist()
    accuracy_batch: list[float] = np.concatenate(accuracy).tolist()

    plot_data(loss_epoch, accuracy_epoch, "Epoch", result_path, plot_type)
    plot_data(loss_batch, accuracy_batch, "Batch", result_path, plot_type)


def save_results(
    results: Results,
    result_path: Path,
) -> None:
    plot_result(results["end_to_end"], "train", result_path / "end2end")
    plot_result(results["end_to_end"], "validation", result_path / "end2end")
    plot_result(results["bridge"], "train", result_path / "bridge")
    plot_result(results["bridge"], "validation", result_path / "bridge")
