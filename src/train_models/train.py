from dataclasses import dataclass
from typing import Union, Literal, Iterator, Optional
from itertools import islice

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn import functional as F
from tqdm import tqdm

from src.types.dataset import Splits
from src.utils.cleanup import cleanup_cache
from src.models.presets import VoiceLMGen
from src.models.voicelm import VoiceLM
from .prepare import (
    get_model,
    get_instruction,
    get_additional,
    get_instruction,
    get_additional,
    get_train_inputs,
    get_true_y,
    get_ignore_token,
    get_accuracy,
    prepare_parameters,
)


@dataclass(frozen=True)
class Result:
    train_loss: list[list[float]]
    train_accuracy: list[list[float]]
    dev_loss: list[list[float]]
    dev_accuracy: list[list[float]]


def create_model(
    model_creator: VoiceLMGen,
) -> Union[VoiceLM, DataParallel[VoiceLM]]:
    model = model_creator()
    prepare_parameters(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model = model.cuda()

    return model


def _train(
    desc_prefix: str,
    model: Union[VoiceLM, DataParallel[VoiceLM]],
    parameters: Iterator[torch.nn.Parameter],
    data_loaders: dict[Splits, DataLoader],
    num_epochs: int,
    max_steps: Optional[int],
    max_new_tokens: int,
    lr: float,
    lr_factor: float,
    patience: int,
) -> Result:
    train_loss = []
    train_accuracy = []
    dev_loss = []
    dev_accuracy = []

    ignore_token = get_ignore_token(model)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_token, reduction="mean")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=patience,
        min_lr=lr * 0.01,
    )

    for epoch in range(num_epochs):
        losses: list[float] = []
        accuracies: list[float] = []
        val_losses: list[float] = []
        val_accuracies: list[float] = []

        d_loader = data_loaders["train"]
        if max_steps is not None:
            d_loader = islice(d_loader, max_steps)

        t = tqdm(
            d_loader,
            desc=f"{desc_prefix} Epoch {epoch + 1}/{num_epochs}",
            total=max_steps,
        )
        model.train()
        for batch in t:
            audio_data: list[np.ndarray] = batch[0]
            transcripts: list[str] = batch[1]

            instruction = get_instruction(model)
            additional = get_additional(model, instruction, transcripts, max_new_tokens)

            audio_inputs = get_train_inputs(model, instruction, additional, audio_data)

            true_y = get_true_y(model, additional)
            predicted_y: Tensor = model(**audio_inputs)  # [batch, seq, vocab]
            predicted_y = F.softmax(predicted_y, dim=-1)  # [batch, seq, vocab]

            if predicted_y.size(1) != true_y.size(1):
                print(
                    f"Warning: predicted_y and true_y have different sequence lengths: {predicted_y.size(1), true_y.size(1)}"
                )
                continue

            loss: Tensor = loss_fn(predicted_y.permute(0, 2, 1), true_y)
            accuracy = get_accuracy(model, predicted_y, true_y)

            losses.append(loss.item())
            accuracies.append(accuracy)

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()

            with torch.no_grad():
                batch = next(iter(data_loaders["dev"]))
                audio_data: list[np.ndarray] = batch[0]
                transcripts: list[str] = batch[1]

                instruction = get_instruction(model)
                additional = get_additional(
                    model, instruction, transcripts, max_new_tokens
                )

                audio_inputs = get_train_inputs(
                    model, instruction, additional, audio_data
                )

                true_y = get_true_y(model, additional)
                predicted_y: Tensor = model(**audio_inputs)  # [batch, seq, vocab]
                predicted_y = F.softmax(predicted_y, dim=-1)  # [batch, seq, vocab]

                if predicted_y.size(1) != true_y.size(1):
                    print(
                        f"Warning for validation: predicted_y and true_y have different sequence lengths: {predicted_y.size(1), true_y.size(1)}"
                    )
                    continue

                loss: Tensor = loss_fn(predicted_y.permute(0, 2, 1), true_y)
                accuracy = get_accuracy(model, predicted_y, true_y)

                val_losses.append(loss.item())
                val_accuracies.append(accuracy)

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


def get_trainable_parameters(model: Union[VoiceLM, DataParallel[VoiceLM]]):
    if isinstance(model, DataParallel):
        return model.module.audio_bridge.bridge_model.parameters()
    return model.audio_bridge.bridge_model.parameters()


def train(
    data_loaders: dict[Splits, DataLoader],
    model_creator: VoiceLMGen,
    num_epochs: int,
    max_steps: Optional[int],
    max_new_tokens: int,
    bridge_lr: float,
    lr_factor: float,
    patience: int,
) -> tuple[Result, VoiceLM]:
    model = create_model(model_creator)
    bridge_results = _train(
        "Bridge",
        model,
        get_trainable_parameters(model),
        data_loaders,
        num_epochs,
        max_steps,
        max_new_tokens,
        bridge_lr,
        lr_factor,
        patience,
    )

    cleanup_cache()

    model = get_model(model)
    model.to("cpu")
    return bridge_results, model
