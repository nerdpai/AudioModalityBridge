import gc
from dataclasses import dataclass
from typing import Union
from itertools import islice

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import DataParallel, functional as F
from tqdm import tqdm

from src.utils.cleanup import cleanup_model
from src.models.presets import VoiceLMGen
from src.models.voicelm import VoiceLM
from .prepare import (
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
    best_loss: float
    best_accuracy: float
    mean_loss: float
    mean_accuracy: float


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


def get_trainable_parameters(model: Union[VoiceLM, DataParallel[VoiceLM]]):
    if isinstance(model, DataParallel):
        return model.module.audio_bridge.bridge_model.parameters()
    return model.audio_bridge.bridge_model.parameters()


def test_preset(
    data_loader: DataLoader,
    model_creator: VoiceLMGen,
    max_steps: int,
    max_new_tokens: int,
    lr: float,
) -> Result:
    losses = []
    accuracies = []

    model = create_model(model_creator)
    ignore_token = get_ignore_token(model)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_token, reduction="mean")
    optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=lr)

    data_iter = iter(data_loader)
    if len(data_loader) > max_steps:
        data_iter = islice(data_iter, max_steps)
    else:
        max_steps = len(data_loader)

    t = tqdm(data_iter, total=max_steps)
    for batch in t:
        audio_data: list[np.ndarray] = batch[0]
        transcripts: list[str] = batch[1]

        instruction = get_instruction(model)
        additional = get_additional(model, instruction, transcripts, max_new_tokens)

        audio_inputs = get_train_inputs(model, instruction, additional, audio_data)

        true_y = get_true_y(model, additional)
        predicted_y: Tensor = model(**audio_inputs)  # [batch, seq, vocab]
        predicted_y = F.softmax(predicted_y, dim=-1)  # [batch, seq, vocab]

        loss: Tensor = loss_fn(predicted_y.permute(0, 2, 1), true_y)
        accuracy = get_accuracy(model, predicted_y, true_y)

        losses.append(loss.item())
        accuracies.append(accuracy)

        t.set_postfix_str(f"Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    cleanup_model(model)
    gc.collect()

    return Result(
        best_loss=min(losses),
        best_accuracy=max(accuracies),
        mean_loss=np.mean(losses).item(),
        mean_accuracy=np.mean(accuracies).item(),
    )
