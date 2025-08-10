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
    save_pickle(model, model_path / "model.pkl")


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

    loss_epoch: list[float] = np.mean(loss, axis=1).tolist()
    accuracy_epoch: list[float] = np.mean(accuracy, axis=1).tolist()
    loss_batch: list[float] = np.concatenate(loss).tolist()
    accuracy_batch: list[float] = np.concatenate(accuracy).tolist()

    # epoch plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_epoch, color="red", label="Loss")
    plt.plot(accuracy_epoch, color="blue", label="Accuracy")
    plt.title(f"{plot_type.capitalize()} Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(result_path / f"{plot_type}_epoch.png")
    plt.close()

    # batch plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_batch, color="red", label="Loss")
    plt.plot(accuracy_batch, color="blue", label="Accuracy")
    plt.title(f"{plot_type.capitalize()} Metrics per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(result_path / f"{plot_type}_batch.png")
    plt.close()


def save_results(
    results: Results,
    result_path: Path,
) -> None:
    plot_result(results["audio"], "train", result_path / "audio")
    plot_result(results["audio"], "validation", result_path / "audio")
    plot_result(results["bridge"], "train", result_path / "bridge")
    plot_result(results["bridge"], "validation", result_path / "bridge")
