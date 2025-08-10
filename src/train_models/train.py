from dataclasses import dataclass
from typing import Union, Literal, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm

from src.types.dataset import Splits
from src.utils.cleanup import cleanup_cache
from src.models.presets import VoiceLMGen
from src.models.voicelm import VoiceLM
from .prepare import (
    get_model,
    get_true_labels,
    get_inputs,
    prepare_bridge_params,
    prepare_audio_model_params,
)


@dataclass(frozen=True)
class Result:
    train_loss: list[list[float]]
    train_accuracy: list[list[float]]
    dev_loss: list[list[float]]
    dev_accuracy: list[list[float]]


Results = dict[Literal["bridge", "audio"], Result]


def create_model(
    model_creator: VoiceLMGen,
) -> Union[VoiceLM, DataParallel[VoiceLM]]:
    model = model_creator()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model = model.cuda()

    return model


def _train(
    model: Union[VoiceLM, DataParallel[VoiceLM]],
    parameters: Iterator[torch.nn.Parameter],
    data_loaders: dict[Splits, DataLoader],
    num_epochs: int,
    lr: float,
    lr_factor: float,
    patience: int,
) -> Result:
    train_loss = []
    train_accuracy = []
    dev_loss = []
    dev_accuracy = []

    loss_fn = torch.nn.MSELoss(reduction="none")
    accuracy_fn = torch.nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=patience,
        min_lr=lr * 0.1,
    )

    for epoch in range(num_epochs):
        losses: list[float] = []
        accuracies: list[float] = []
        val_losses: list[float] = []
        val_accuracies: list[float] = []

        t = tqdm(data_loaders["train"], desc=f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        for batch in t:
            audio_data: list[np.ndarray] = batch[0]
            transcripts: list[str] = batch[1]

            true_y = get_true_labels(transcripts, model)
            inputs = get_inputs(audio_data, model)
            predicted_y: Tensor = model(**inputs.asdict())

            loss: Tensor = loss_fn(predicted_y, true_y)
            loss = loss.sum(dim=1).mean()
            accuracy: Tensor = accuracy_fn(predicted_y, true_y)
            accuracy = accuracy.mean()

            losses.append(loss.item())
            accuracies.append(accuracy.item())

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()

            with torch.no_grad():
                batch = next(iter(data_loaders["dev"]))
                audio_data: list[np.ndarray] = batch[0]
                transcripts: list[str] = batch[1]

                true_y = get_true_labels(transcripts, model)
                inputs = get_inputs(audio_data, model)
                predicted_y: Tensor = model(**inputs.asdict())

                loss: Tensor = loss_fn(predicted_y, true_y)
                loss = loss.sum(dim=1).mean()
                accuracy: Tensor = accuracy_fn(predicted_y, true_y)
                accuracy = accuracy.mean()

                val_losses.append(loss.item())
                val_accuracies.append(accuracy.item())

            t.set_postfix_str(
                f"Loss: {losses[-1]:.2f}, "
                f"Accuracy: {accuracies[-1]:.2f}, "
                f"Validation: ({val_losses[-1]:.2f}, {val_accuracies[-1]:.2f})"
            )

        train_loss.append(losses)
        train_accuracy.append(accuracies)
        dev_loss.append(val_losses)
        dev_accuracy.append(val_accuracies)

    return Result(train_loss, train_accuracy, dev_loss, dev_accuracy)


def train(
    data_loaders: dict[Splits, DataLoader],
    model_creator: VoiceLMGen,
    num_epochs: int,
    bridge_lr: float,
    audio_lr: float,
    lr_factor: float,
    patience: int,
) -> tuple[Results, VoiceLM]:
    model = create_model(model_creator)
    bridge_params = prepare_bridge_params(model)
    bridge_results = _train(
        model,
        bridge_params,
        data_loaders,
        num_epochs,
        bridge_lr,
        lr_factor,
        patience,
    )

    cleanup_cache()

    audio_params = prepare_audio_model_params(model)
    audio_results = _train(
        model,
        audio_params,
        data_loaders,
        num_epochs,
        audio_lr,
        lr_factor,
        patience,
    )

    results: Results = {
        "bridge": bridge_results,
        "audio": audio_results,
    }
    model = get_model(model)
    model.to("cpu")
    return results, model
