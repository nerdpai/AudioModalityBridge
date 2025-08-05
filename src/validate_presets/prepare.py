from typing import Union

import numpy as np
import torch
from torch.nn import DataParallel

from src.models.voicelm import VoiceLM


def get_model(model: Union[VoiceLM, DataParallel[VoiceLM]]) -> VoiceLM:
    if isinstance(model, DataParallel):
        return model.module
    return model


@torch.no_grad()
def get_true_labels(
    transcripts: list[str], model: Union[DataParallel[VoiceLM], VoiceLM]
) -> torch.Tensor:
    voicelm = get_model(model)

    tokenized = voicelm.tokenizer(
        transcripts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )

    input_ids: torch.Tensor = tokenized["input_ids"]  # type: ignore
    attention_mask: torch.Tensor = tokenized["attention_mask"]  # type: ignore

    return model(
        input_ids=input_ids.to(voicelm.device),
        attention_mask=attention_mask.to(voicelm.device),
    )


@torch.no_grad()
def get_inputs(
    audio_samples: list[np.ndarray], model: Union[DataParallel[VoiceLM], VoiceLM]
):
    model = get_model(model)
    return model.audio_bridge.preprocess_audio(audio_samples)


def prepare_parameters(model: VoiceLM):
    for p in model.language_model.parameters():
        p.requires_grad = False

    for p in model.audio_bridge.audio_model.parameters():
        p.requires_grad = False

    for p in model.audio_bridge.bridge_model.parameters():
        p.requires_grad = True
