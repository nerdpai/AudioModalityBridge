import random
from typing import Optional, Union, TypeAlias

import numpy as np
import torch
from torch.nn import DataParallel
from transformers.tokenization_utils_base import BatchEncoding

from src.constants.few_shot import FEW_SHOT_TEMPLATES
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

    [language] = random.sample(list(FEW_SHOT_TEMPLATES.keys()), 1)
    template = FEW_SHOT_TEMPLATES[language]
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
    eos_token: str = voicelm.tokenizer.eos_token  # type: ignore
    instructions = [instruction] * len(transcripts)

    inputs = []
    for instruction, transcript in zip(instructions, transcripts):
        inputs.append([instruction, transcript, eos_token])

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
def remove_last_token(model: Model, additional: list[str]) -> list[str]:
    voicelm = get_model(model)
    tokenizer = voicelm.tokenizer

    tokenized = [
        tokenizer.encode(
            text, add_special_tokens=False, padding=False, truncation=False
        )
        for text in additional
    ]
    truncated = [tokens[:-1] for tokens in tokenized]
    decoded = [
        tokenizer.decode(tokens, skip_special_tokens=False) for tokens in truncated
    ]

    return decoded


@torch.no_grad()
def get_accuracy(
    model: Model,
    predicted_y: torch.Tensor,  # [batch, seq, vocab]
    true_y: torch.Tensor,  # [batch, seq]
) -> float:
    predicted_labels = predicted_y.argmax(dim=-1)
    correct = (predicted_labels == true_y).float().sum()
    total = true_y.numel()
    return (correct / total).item()


@torch.no_grad()
def get_true_y(
    model: Model,
    additional: list[str],
) -> torch.Tensor:
    ids, _ = get_ids(model, additional)
    return ids


@torch.no_grad()
def get_ignore_token(model: Model) -> int:
    voicelm = get_model(model)
    return voicelm.tokenizer.pad_token_id  # type: ignore


@torch.no_grad()
def get_train_inputs(
    model: Model,
    instruction: str,
    additional: list[str],
    input_data: list[np.ndarray],
) -> dict[str, Optional[torch.Tensor]]:
    voicelm = get_model(model)

    eos_token: str = voicelm.tokenizer.eos_token  # type: ignore
    additional = remove_last_token(voicelm, additional)
    additional = [
        eos_token + text for text in additional
    ]  # add eos_token to the beginning, after lm_head it will predict first token from additional

    instruction_ids, instruction_mask = get_ids(
        voicelm, [instruction] * len(input_data)
    )
    additional_ids, additional_mask = get_ids(voicelm, additional)

    preprocessed = get_audio_preprocessed(voicelm, input_data)
    audio_inputs = preprocessed.audio_inputs
    chunk_mask = preprocessed.chunk_mask
    attention_mask = preprocessed.attention_mask

    return {
        "input_ids": None,
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
