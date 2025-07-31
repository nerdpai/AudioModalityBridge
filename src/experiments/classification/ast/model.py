import numpy as np
import torch
from scipy.io.wavfile import read
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

feature_extractor = ASTFeatureExtractor.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="1913c7d7957c8f5c1e32f20f168a3d12ec9530fb",
)

audio = read("src/experiments/data/war_face.wav")[1]
audio = np.asarray(audio, dtype=np.float32)

print(f"Audio shape: {audio.shape}")
input_values = feature_extractor(
    audio, return_tensors="pt", sampling_rate=16000
).input_values

input_values = input_values.to(dtype=torch.float32)

output = model(input_values.to(model.device)).logits
output = output.argmax(dim=-1).item()

output = model.config.id2label[output]

print(output)
