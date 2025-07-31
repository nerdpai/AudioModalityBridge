import numpy as np
import torch
from scipy.io.wavfile import read
from transformers.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)

from src.constants.secrets import SECRETS

MODEL_ID = "openai/whisper-small.en"

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="e8727524f962ee844a7319d92be39ac1bd25655a",
    torch_dtype=torch.float32,
    device_map="cuda",
)

encoder = model.model.encoder

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="e8727524f962ee844a7319d92be39ac1bd25655a",
)

sr, audio = read("src/experiments/data/war_face.wav")
audio = np.asarray(audio, dtype=np.float32)

print(f"Audio shape: {audio.shape}, Sampling rate: {sr}")

input_features = feature_extractor(
    audio,
    return_tensors="pt",
    sampling_rate=16000,
).input_features


output = encoder.forward(
    input_features=input_features.to(model.device),
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
)

print(output.last_hidden_state.shape)
