from typing import Protocol, Literal, Optional
from dataclasses import dataclass, asdict

import numpy as np
from transformers.feature_extraction_utils import BatchFeature
import torch
from torch import Tensor
from torch import nn

from src.constants.dataset import SAMPLING_RATE
from .modal_translator import ModalTranslator
from .audio_model import AudioModel

TensorTypes = Literal["pt"]


@dataclass
class PreProcessed:
    inputs: Tensor
    attention_mask: Optional[Tensor]
    chunk_mask: Tensor

    def asdict(self):
        return asdict(self)


class FeatureExtractor(Protocol):
    def __call__(
        self,
        raw_speech: list[np.ndarray],
        return_tensors: TensorTypes,
        sampling_rate: int,
        return_attention_mask: bool,
        padding: bool,
        **kwargs,
    ) -> BatchFeature: ...


class AudioBridge(nn.Module):
    def __init__(
        self,
        audio_model: AudioModel,
        bridge_model: ModalTranslator,
        sampling_rate: int = SAMPLING_RATE,
        max_processing_time: float = 30,
        min_processing_time: float = 1,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.bridge_model = bridge_model
        self.feature_extractor: FeatureExtractor = audio_model.feature_extractor  # type: ignore
        self.feature_property_name = audio_model.config.feature_property_name

        self.sr = sampling_rate
        self.max_processing_sr = int(max_processing_time * self.sr)
        self.min_processing_sr = int(min_processing_time * self.sr)

        self.max_processing_seq = int(max_processing_time * self.bridge_model.in_seq)
        self.min_processing_seq = int(min_processing_time * self.bridge_model.in_seq)

    @property
    def device(self) -> torch.device:
        device = getattr(self.audio_model, "device", None)
        if device is None:
            device = torch.device("cpu")
        return device

    # PreProcess ----------------------

    def chunk_audio(
        self, audio_array: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        chunks = []
        for i in range(0, len(audio_array), self.max_processing_sr):
            chunk = audio_array[i : i + self.max_processing_sr]
            chunks.append(chunk)

        if len(chunks) > 0 and len(chunks[-1]) < self.min_processing_sr:
            chunks.pop()

        if len(chunks) > 0 and len(chunks[-1]) < self.max_processing_sr:
            pad_len = self.max_processing_sr - len(chunks[-1])
            chunks[-1] = np.pad(
                chunks[-1],
                (0, pad_len),
                "constant",
            )

        attention_masks = []
        for chunk in chunks:
            flipped = np.flip(chunk, axis=-1)
            nonzero_mask = flipped != 0

            if nonzero_mask.any():
                nonzero_pos = np.argmax(nonzero_mask)
                nonzero_pos = len(flipped) - nonzero_pos
                attention_mask = np.zeros_like(flipped, dtype=np.int32)
                attention_mask[:nonzero_pos] = 1
            else:
                attention_mask = np.zeros_like(flipped, dtype=np.int32)

            attention_masks.append(attention_mask)

        return chunks, attention_masks

    def get_features(self, batch_chunks: list[list[np.ndarray]]) -> list[Tensor]:
        features = []
        for chunks in batch_chunks:
            feature = self.feature_extractor(
                chunks,
                return_tensors="pt",
                sampling_rate=self.sr,
                return_attention_mask=False,
                padding=False,
            )

            input_values = feature[self.feature_property_name]
            features.append(input_values)

        return features

    @torch.no_grad()
    def prepare_model_inputs(
        self,
        features: list[Tensor],
        inputs_dtype: torch.dtype,
        attention_masks: list[list[np.ndarray]],
    ) -> PreProcessed:
        inputs = torch.stack(features).to(dtype=inputs_dtype)
        attention_mask = torch.tensor(np.array(attention_masks), dtype=torch.int32)
        chunk_mask = attention_mask.any(dim=-1).to(torch.int32)

        inputs = inputs.to(self.device)
        attention_mask = attention_mask.to(self.device)
        chunk_mask = chunk_mask.to(self.device)

        if inputs.size() != attention_mask.size():
            attention_mask = None

        return PreProcessed(
            inputs,  # [batch, chunks, max_len]
            attention_mask,  # [batch, chunks, max_len]
            chunk_mask,  # # [batch, chunks]
        )

    def pad_audio(
        self,
        audio: list[np.ndarray],
    ) -> list[np.ndarray]:
        max_len = max(len(a) for a in audio)
        return [
            np.pad(a, (0, max_len - len(a)), mode="constant", constant_values=0)
            for a in audio
        ]

    def preprocess_audio(
        self,
        audio: list[np.ndarray],  # [batch, samples]
    ) -> PreProcessed:
        audio = self.pad_audio(audio)

        batch_chunks = []
        batch_masks = []
        for audio_sample in audio:
            chunks, attention_masks = self.chunk_audio(audio_sample)
            if len(chunks) == 0:
                raise ValueError("Audio sample is too short to process.")
            batch_chunks.append(chunks)
            batch_masks.append(attention_masks)

        features = self.get_features(batch_chunks)  # [batch, chunks, features]

        return self.prepare_model_inputs(
            features,
            self.bridge_model.torch_dtype,
            batch_masks,
        )

    # Forward ----------------------

    @torch.no_grad()
    def create_embed_attention_mask(
        self,
        audio_embed: Tensor,  # [batch, chunks, seq, hidden]
        chunk_mask: Tensor,  # [batch, chunks]
    ) -> Tensor:
        """This will work for wav2vec2 like models."""
        last_embeddings = audio_embed[:, :, -1:, :]  # [batch, chunks, 1, hidden]

        attention_mask = torch.eq(audio_embed, last_embeddings).all(
            dim=-1
        )  # [batch, chunks, seq]

        attention_mask = attention_mask.to(torch.int32)
        mask_sum = attention_mask.sum(dim=-1, keepdim=True)  # [batch, chunks, 1]
        is_only_last = (mask_sum == 1).to(torch.int32)  # [batch, chunks, 1]

        attention_mask = attention_mask * (1 - is_only_last)
        attention_mask = (attention_mask == 0).to(torch.int32)
        attention_mask = attention_mask * chunk_mask.unsqueeze(-1)
        return attention_mask

    def additional_padding(
        self,
        audio_embed: Tensor,  # [batch, chunks, seq, hidden]
    ) -> Tensor:
        if audio_embed.size(2) < self.max_processing_seq:
            last_token = audio_embed[:, :, -1:, :]  # [batch, chunks, 1, hidden]
            pad_len = self.max_processing_seq - audio_embed.size(2)
            audio_embed = torch.cat(
                [
                    audio_embed,
                    last_token.repeat(1, 1, pad_len, 1),
                ],
                dim=2,
            )

        return audio_embed

    def forward(
        self,
        inputs: Tensor,  # [batch, chunks, max_len...]
        attention_mask: Optional[Tensor],  # [batch, chunks, max_len...]
        chunk_mask: Tensor,  # [batch, chunks]
    ) -> tuple[Tensor, Tensor]:
        input_size = inputs.size()
        reshape = (
            -1,  # batch * chunks
            *input_size[2:],  # max_len
        )
        inputs = inputs.view(reshape)
        if attention_mask is not None:
            attention_mask = attention_mask.view(reshape)

        audio_embed: Tensor = self.audio_model(
            inputs,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state  # [batch * chunks, seq, hidden]

        embed_size = audio_embed.size()
        reshape = (
            input_size[0],  # batch
            input_size[1],  # chunks
            embed_size[1],  # seq
            embed_size[2],  # hidden
        )
        audio_embed = audio_embed.view(reshape)
        audio_embed = self.additional_padding(audio_embed)

        embed_mask = self.create_embed_attention_mask(
            audio_embed, chunk_mask
        )  # [batch, chunks, seq]

        audio_embed = audio_embed.view(
            input_size[0],  # batch
            -1,  # chunks * seq
            embed_size[2],  # hidden
        )
        embed_mask = embed_mask.view(
            input_size[0],  # batch
            -1,  # chunks * seq
        )

        return self.bridge_model(audio_embed, embed_mask)
