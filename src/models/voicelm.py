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

    @torch.no_grad()
    def tokenize_preprocessed(
        self, text: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]  # type: ignore

    def gather(
        self,
        embeds: torch.Tensor,  # [batch, seq, hidden]
        attention: Optional[torch.Tensor],  # [batch, seq]
        arange: torch.Tensor,  # [batch, seq]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        embeds = torch.gather(
            embeds,
            dim=1,
            index=arange.unsqueeze(-1).expand(-1, -1, embeds.shape[-1]),
        )
        if attention is not None:
            attention = torch.gather(attention, dim=1, index=arange)  # type: ignore

        return embeds, attention

    @torch.no_grad()
    def gather_mask(
        self,
        mask1: torch.Tensor,  # [batch, seq]
        mask2: torch.Tensor,  # [batch, seq]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_all = torch.cat([mask1, mask2], dim=1)
        arange = torch.arange(mask_all.shape[1], device=mask_all.device)  # [seq]
        arange = arange.unsqueeze(0)  # [1, seq]
        arange = arange.expand(mask_all.shape[0], -1)  # [batch, seq]

        cleanup_arrange = arange

        pad1_len = (1 - mask1).sum(dim=1)  # [batch]
        pad1_len = pad1_len.unsqueeze(1)  # [batch, 1]
        pad1_len = pad1_len.expand(-1, mask_all.shape[1])  # [batch, seq]

        alter_mask_all = 1 - mask_all  # [batch, seq]
        arange = arange + pad1_len * alter_mask_all  # good but possible out of bounds

        num_tokens = mask1.sum(dim=1, keepdim=True) + mask2.sum(
            dim=1, keepdim=True
        )  # [batch, 1]
        new_mask = cleanup_arrange < num_tokens  # [batch, seq]
        new_mask = new_mask.to(torch.int32)

        arange = arange * new_mask  # [batch, seq]

        return arange, new_mask

    @torch.no_grad()
    def gather_rearange(
        self,
        instruction_mask: torch.Tensor,  # [batch, seq]
        additional_mask: torch.Tensor,  # [batch, seq]
        attention_mask: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        right_len = additional_mask.shape[1] + attention_mask.shape[1]
        arange = torch.arange(right_len, device=attention_mask.device)  # [seq]
        arange = arange.unsqueeze(0)  # [1, seq]
        arange = arange.expand(instruction_mask.shape[0], -1)  # [batch, seq]
        arange = arange + attention_mask.sum(dim=1, keepdim=True)  # [batch, seq]
        arange = arange + instruction_mask.shape[1]  # [batch, seq]

        left_arange = torch.arange(
            instruction_mask.shape[1],
            device=attention_mask.device,
        )  # [seq]
        left_arange = left_arange.unsqueeze(0)  # [1, seq]
        left_arange = left_arange.expand(instruction_mask.shape[0], -1)  # [batch, seq]
        arange = torch.cat([left_arange, arange], dim=1)

        valid_len = instruction_mask.sum(dim=1) \
            + additional_mask.sum(dim=1) # fmt: skip

        mask_arange = torch.arange(arange.shape[1], device=arange.device)  # [seq]
        mask_arange = mask_arange.unsqueeze(0)  # [1, seq]
        mask_arange = mask_arange.expand(arange.shape[0], -1)

        mask = mask_arange < valid_len.unsqueeze(1)  # [batch, seq]
        mask = mask.to(torch.int32)
        arange = arange * mask  # [batch, seq]

        return arange

    def add_eos_token(
        self,
        input_embeds: torch.Tensor,  # [batch, seq, hidden]
        attention_mask: torch.Tensor,  # [batch, seq]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            eos_token: str = self.tokenizer.eos_token  # type: ignore
            eos_token_id, _ = self.tokenize_preprocessed([eos_token])  # [1, 1]
            eos_token_embed = self.language_model.embed_layer(
                eos_token_id.to(self.device)
            )  # [1, 1, hidden]
            eos_token_embed = eos_token_embed.expand(
                input_embeds.shape[0], 1, -1
            )  # [batch, 1, hidden]

            arange = torch.arange(
                input_embeds.shape[1] + 1, device=self.device
            )  # [seq]
            arange = arange.unsqueeze(0)  # [1, seq]
            arange = arange.expand(input_embeds.shape[0], -1)  # [batch, seq]

            last_pos = attention_mask.sum(dim=1, keepdim=True)  # [batch, 1]
            arange = torch.where(
                arange == last_pos,
                input_embeds.shape[1],
                arange,
            )
            arange[:, -1] = last_pos.squeeze(-1)

        input_embeds = torch.cat([input_embeds, eos_token_embed], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1), device=self.device, dtype=torch.int32
                ),
            ],
            dim=1,
        )

        input_embeds, attention_mask = self.gather(
            input_embeds, attention_mask, arange
        )  # type: ignore

        return input_embeds, attention_mask

    def forward(
        self,
        instruction_ids: torch.LongTensor,
        instruction_mask: torch.LongTensor,
        additional_ids: torch.LongTensor,
        additional_mask: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        audio_inputs: Optional[torch.FloatTensor] = None,
        attention_mask: torch.Tensor = None,  # type: ignore
        chunk_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            ValueError("attention_mask must be provided.")

        instruction_embeds = self.language_model.embed_layer(instruction_ids)
        additional_embeds = self.language_model.embed_layer(additional_ids)

        if input_ids is not None:
            input_embeds = self.language_model.embed_layer(
                input_ids
            )  # [batch, seq, hidden]
        elif audio_inputs is not None:
            input_embeds, attention_mask = self.audio_bridge(
                audio_inputs,
                attention_mask,
                chunk_mask,
            )
        else:
            raise ValueError("Either input_ids or audio_samples must be provided.")

        input_embeds, attention_mask = self.add_eos_token(input_embeds, attention_mask)

        rearange = self.gather_rearange(
            instruction_mask, additional_mask, attention_mask
        )
        arange, attention_mask = self.gather_mask(attention_mask, additional_mask)
        input_embeds = torch.cat([input_embeds, additional_embeds], dim=1)
        input_embeds, _ = self.gather(input_embeds, None, arange)

        input_embeds = torch.cat([instruction_embeds, input_embeds], dim=1)
        attention_mask = torch.cat([instruction_mask, attention_mask], dim=1)

        last_hidden = self.language_model(
            input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        last_hidden, _ = self.gather(last_hidden, None, rearange)
        last_hidden = last_hidden[
            :,
            instruction_mask.shape[1] : instruction_mask.shape[1]
            + additional_mask.shape[1],
            :,
        ]
        return last_hidden

    @torch.no_grad()
    def process_text_input(
        self, message: Union[str, dict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(message, dict):
            message = self.tokenizer.apply_chat_template(message, tokenize=False)  # type: ignore

        input_ids, attention_mask = self.tokenize_preprocessed([message])  # type: ignore

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        input_embeds = self.language_model.embed_layer(input_ids)

        return input_embeds, attention_mask

    @torch.no_grad()
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
    def pad_seq(
        self,
        embeds_batch: list[
            torch.Tensor
        ],  # [batch, 1, seq*, hidden] *all seq are different
        masks_batch: list[torch.Tensor],  # [batch, 1, seq*] *all seq are different
        pad_token: torch.Tensor,  # [1, 1, hidden]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_seq_length = max(mask.shape[1] for mask in masks_batch)
        batch_size = len(embeds_batch)
        hidden_size = embeds_batch[0].shape[-1]

        padded_embeds = torch.zeros(
            (batch_size, max_seq_length, hidden_size),
            dtype=embeds_batch[0].dtype,
            device=self.device,
        )  # [batch, seq, hidden]
        padded_masks = torch.zeros(
            (batch_size, max_seq_length),
            dtype=torch.int32,
            device=self.device,
        )  # [batch, seq]

        for i, (embeds, mask) in enumerate(zip(embeds_batch, masks_batch)):
            seq_length = embeds.shape[1]
            padded_embeds[i, :seq_length, :] = embeds.squeeze(0)
            padded_masks[i, :seq_length] = mask.squeeze(0)

        pad_token = pad_token.expand(
            batch_size, max_seq_length, -1
        )  # [batch, seq, hidden]
        padded_embeds = padded_embeds + pad_token * (1 - padded_masks.unsqueeze(-1))

        return padded_embeds, padded_masks  # [batch, seq, (hidden)]

    @torch.no_grad()
    def generate(self, batch: list[list[Union[np.ndarray, str, dict]]], **kwargs):
        embeds_batch = []  # [batch, 1, seq, hidden]
        masks_batch = []  # [batch, 1, seq]

        for item in batch:
            embeds_item: list[torch.Tensor] = []
            masks_item: list[torch.Tensor] = []
            for data in item:
                if isinstance(data, dict) or isinstance(data, str):
                    embeds, mask = self.process_text_input(data)  # [1, seq, hidden]
                elif isinstance(data, np.ndarray):
                    embeds, mask = self.process_audio_input(data)  # [1, seq, hidden]
                else:
                    raise ValueError("Input must be either a message or a numpy array.")

                embeds_item.append(embeds)
                masks_item.append(mask)

            item_embeds = torch.cat(embeds_item, dim=1)
            item_mask = torch.cat(masks_item, dim=1)

            embeds_batch.append(item_embeds)
            masks_batch.append(item_mask)

        pad_token: str = self.language_model.tokenizer.pad_token  # type: ignore
        pad_token_id, _ = self.tokenize_preprocessed([pad_token])  # [1, 1]
        pad_token_embed = self.language_model.embed_layer(
            pad_token_id.to(self.device)
        )  # [1, 1, hidden]

        input_embeds, attention_mask = self.pad_seq(
            embeds_batch, masks_batch, pad_token_embed
        )

        return self.language_model.generate(
            inputs_embeds=input_embeds,  # type: ignore
            attention_mask=attention_mask,  # type: ignore
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
