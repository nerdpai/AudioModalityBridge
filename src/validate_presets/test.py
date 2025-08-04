import gc
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tqdm import tqdm

from src.models.presets import VoiceLMGen
from src.models.voicelm import VoiceLM
from .prepare import get_true_labels, get_inputs, prepare_parameters
from .clean import cleanup_model


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


def get_model(model: Union[VoiceLM, DataParallel[VoiceLM]]) -> VoiceLM:
    if isinstance(model, DataParallel):
        return model.module
    return model


def test_preset(
    data_loader: DataLoader,
    model_creator: VoiceLMGen,
    lr: float,
) -> Result:
    losses = []
    accuracies = []

    model = create_model(model_creator)

    loss_fn = torch.nn.MSELoss(reduction="none")
    accuracy_fn = torch.nn.CosineSimilarity(dim=1)
    optimizer = torch.optim.Adam(get_trainable_parameters(model), lr=lr)

    t = tqdm(data_loader)
    for batch in t:
        audio_data: list[np.ndarray] = batch[0]
        transcripts: list[str] = batch[1]

        true_y = get_true_labels(transcripts, get_model(model))
        inputs = get_inputs(audio_data, get_model(model))
        predicted_y: Tensor = model(**inputs.asdict())

        loss: Tensor = loss_fn(predicted_y, true_y)
        loss = loss.sum(dim=1)
        accuracy: Tensor = accuracy_fn(predicted_y, true_y)

        losses.append(loss.mean().item())
        accuracies.append(accuracy.mean().item())

        t.set_postfix_str(
            f"Loss: {np.mean(losses[-1]):.4f}, Accuracy: {np.mean(accuracies[-1]):.4f}"
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    cleanup_model(model)
    gc.collect()

    return Result(
        best_loss=min(losses),
        best_accuracy=min(accuracies),
        mean_loss=np.mean(losses).item(),
        mean_accuracy=np.mean(accuracies).item(),
    )
