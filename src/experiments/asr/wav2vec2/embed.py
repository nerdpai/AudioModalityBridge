import numpy as np
import torch
from scipy.io.wavfile import read
from transformers.models.wav2vec2 import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)

from src.constants.secrets import SECRETS

MODEL_ID = "facebook/wav2vec2-large-960h"

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="bdeaacdf88f7a155f50a2704bc967aa81fbbb2ab",
    torch_dtype=torch.float32,
    device_map="cuda",
)

encoder = model.wav2vec2

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="bdeaacdf88f7a155f50a2704bc967aa81fbbb2ab",
)

sr, audio = read("src/experiments/data/war_face.wav")
audio = np.asarray(audio, dtype=np.float32)

print(f"Audio shape: {audio.shape}, Sampling rate: {sr}")

input_values = processor(
    audio,
    return_tensors="pt",
    sampling_rate=16000,
).input_values


output = encoder.forward(
    input_values=input_values.to(model.device),
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
)

print(output.last_hidden_state.shape)
