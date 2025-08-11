import numpy as np
import torch
from torchaudio import load
from transformers.models.audio_spectrogram_transformer import (
    ASTForAudioClassification,
    ASTFeatureExtractor,
)

from src.constants.secrets import SECRETS


MODEL_ID = "aicinema69/audio-emotion-detector-try2"

model = ASTForAudioClassification.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="1913c7d7957c8f5c1e32f20f168a3d12ec9530fb",
    torch_dtype=torch.float32,
    device_map="cuda",
)
model = model.audio_spectrogram_transformer

feature_extractor = ASTFeatureExtractor.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="1913c7d7957c8f5c1e32f20f168a3d12ec9530fb",
)

audio, sr = load(
    "src/experiments/data/war_face.wav",
    normalize=True,
    channels_first=True,
    backend="ffmpeg",
)
audio = audio[0].numpy()
audio = np.asarray(audio, dtype=np.float32)

print(f"Audio shape: {audio.shape}, Sampling rate: {sr}")
input_values = feature_extractor(
    audio, return_tensors="pt", sampling_rate=16000, return_attention_mask=True
).input_values

print(f"Input values shape: {input_values.shape}")

input_values = input_values.to(dtype=torch.float32)

output = model.forward(input_values.to(model.device))

print(output.last_hidden_state.shape)
