from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from .audio_bridge import AudioBridge
from .language_model import LanguageModel


class VoiceLM(nn.Module):
    def __init__(
        self,
        audio_bridge: AudioBridge,
        language_model: LanguageModel,
    ):
        super(VoiceLM, self).__init__()

        self.audio_bridge = audio_bridge
        self.language_model = language_model

    @property
    def device(self) -> torch.device:
        return self.language_model.device

    @property
    def tokenizer(self):
        return self.language_model.tokenizer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_samples: Optional[list[np.ndarray]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if input_ids is not None:
            input_embeds = self.language_model.embed_layer(input_ids)
        elif audio_samples is not None:
            preprocessed = self.audio_bridge.preprocess_audio(audio_samples)
            input_embeds, attention_mask = self.audio_bridge(**preprocessed.asdict())
        else:
            raise ValueError("Either input_ids or audio_samples must be provided.")

        last_tokens = self.language_model(
            input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        return last_tokens

    def process_text_input(
        self, message: Union[str, dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(message, dict):
            message = self.tokenizer.apply_chat_template(message, tokenize=False)  # type: ignore

        tokenized = self.tokenizer(
            message,  # type: ignore
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
        )
        input_ids: torch.Tensor = tokenized["input_ids"]  # type: ignore
        attention_mask: torch.Tensor = tokenized["attention_mask"]  # type: ignore

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        input_embeds = self.language_model.embed_layer(input_ids)

        return input_embeds, attention_mask

    def process_audio_input(
        self, audio_samples: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preprocessed = self.audio_bridge.preprocess_audio([audio_samples])

        input_embeds, attention_mask = self.audio_bridge(**preprocessed.asdict())

        seq_lengths = attention_mask.sum(dim=1)  # [batch]
        min_seq_length = seq_lengths.min().item()

        input_embeds = input_embeds[:, :min_seq_length, :]
        attention_mask = attention_mask[:, :min_seq_length]

        return input_embeds, attention_mask

    @torch.no_grad()
    def generate(self, inputs: list[Union[np.ndarray, str, dict]], **kwargs):
        embeds_list = []  # [parts, batch, seq, hidden]
        masks_list = []  # [parts, batch, seq]

        for input_data in inputs:
            if isinstance(input_data, dict) or isinstance(input_data, str):
                embeds, mask = self.process_text_input(input_data)
            elif isinstance(input_data, np.ndarray):
                embeds, mask = self.process_audio_input(input_data)
            else:
                raise ValueError("Input must be either a string or a numpy array.")

            embeds_list.append(embeds)
            masks_list.append(mask)

        input_embeds = torch.cat(embeds_list, dim=1)
        attention_mask = torch.cat(masks_list, dim=1)

        self.language_model.generate(
            inputs_embeds=input_embeds,  # type: ignore
            attention_mask=attention_mask,  # type: ignore
            **kwargs,
        )
