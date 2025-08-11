import numpy as np
import torch
from torchaudio import load
from transformers.models.speech_to_text import (
    Speech2TextFeatureExtractor,
    Speech2TextForConditionalGeneration,
)

from src.constants.secrets import SECRETS

MODEL_ID = "facebook/s2t-large-librispeech-asr"

model = Speech2TextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="a4b4750ad1425acda0dbd1daa9188d5fd7872491",
    torch_dtype=torch.float32,
    device_map="cuda",
)

encoder = model.model.encoder

feature_extractor = Speech2TextFeatureExtractor.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="a4b4750ad1425acda0dbd1daa9188d5fd7872491",
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
