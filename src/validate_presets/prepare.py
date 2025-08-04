import numpy as np
import torch

from src.models.voicelm import VoiceLM


@torch.no_grad()
def get_true_labels(transcripts: list[str], model: VoiceLM) -> torch.Tensor:
    tokenized = model.tokenizer(
        transcripts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )

    input_ids: torch.Tensor = tokenized["input_ids"]  # type: ignore
    attention_mask: torch.Tensor = tokenized["attention_mask"]  # type: ignore

    return model(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
    )


def prepare_parameters(model: VoiceLM):
    for p in model.language_model.parameters():
        p.requires_grad = False

    for p in model.audio_bridge.audio_model.parameters():
        p.requires_grad = False

    for p in model.audio_bridge.bridge_model.parameters():
        p.requires_grad = True
