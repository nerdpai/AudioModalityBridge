import random
from typing import Optional, Union, TypeAlias

import numpy as np
import torch
from torch.nn import DataParallel
from transformers.tokenization_utils_base import BatchEncoding

from src.constants.template import GENERATION_TEMPLATE
from src.models.voicelm import VoiceLM

Model: TypeAlias = Union[DataParallel[VoiceLM], VoiceLM]


def get_model(model: Model) -> VoiceLM:
    if isinstance(model, DataParallel):
        return model.module
    return model


@torch.no_grad()
def get_ids(
    model: Model,
    text: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    voicelm = get_model(model)

    encodings: BatchEncoding = voicelm.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_attention_mask=True,
        add_special_tokens=False,
    )

    input_ids: torch.Tensor = encodings["input_ids"]  # [1, seq_len]
    attention_mask: torch.Tensor = encodings["attention_mask"]  # [1, seq_len]

    return input_ids.to(voicelm.device), attention_mask.to(voicelm.device)


def get_instruction(model: Model) -> str:
    """Language instruction without eot_id in the end of template."""

    voicelm = get_model(model)
    eos_token: str = voicelm.tokenizer.eos_token  # type: ignore

    template = GENERATION_TEMPLATE
    instruction: str = voicelm.tokenizer.apply_chat_template(template, tokenize=False)  # type: ignore
    return instruction.removesuffix(eos_token)


@torch.no_grad()
def get_additional(
    model: Model,
    instruction: str,
    transcripts: list[str],
    max_new_tokens: int,
) -> list[str]:
    voicelm = get_model(model)
    instructions = [instruction] * len(transcripts)

    inputs = []
    for instruction, transcript in zip(instructions, transcripts):
        inputs.append([instruction, transcript])

    additional_ids: torch.Tensor = voicelm.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=None,
    )  # type: ignore
    additional: list[str] = voicelm.tokenizer.batch_decode(
        additional_ids, skip_special_tokens=False
    )

    return additional


@torch.no_grad()
def get_audio_preprocessed(model: Model, audio_samples: list[np.ndarray]):
    model = get_model(model)
    return model.audio_bridge.preprocess_audio(audio_samples)


@torch.no_grad()
def get_model_inputs(
    model: Model,
    instruction: str,
    additional: list[str],
    input_data: Union[list[np.ndarray], list[str]],
) -> dict[str, Optional[torch.Tensor]]:
    voicelm = get_model(model)

    instruction_ids, instruction_mask = get_ids(
        voicelm, [instruction] * len(input_data)
    )
    additional_ids, additional_mask = get_ids(voicelm, additional)

    input_ids = None
    audio_inputs = None
    chunk_mask = None
    if isinstance(input_data[0], str):
        input_ids, attention_mask = get_ids(voicelm, input_data)
    else:
        preprocessed = get_audio_preprocessed(voicelm, input_data)
        audio_inputs = preprocessed.audio_inputs
        chunk_mask = preprocessed.chunk_mask
        attention_mask = preprocessed.attention_mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "instruction_ids": instruction_ids,
        "instruction_mask": instruction_mask,
        "additional_ids": additional_ids,
        "additional_mask": additional_mask,
        "audio_inputs": audio_inputs,
        "chunk_mask": chunk_mask,
    }


def prepare_parameters(model: VoiceLM):
    for p in model.language_model.parameters():
        p.requires_grad = False

    for p in model.audio_bridge.audio_model.parameters():
        p.requires_grad = False

    for p in model.audio_bridge.bridge_model.parameters():
        p.requires_grad = True
