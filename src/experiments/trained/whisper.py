from pathlib import Path

import numpy as np
from torchaudio import load

from src.utils.model_io import load_torch
from src.models.voicelm import VoiceLM

MODEL_PATH = Path("./.models/asr/whisper/model.pt").resolve()
model = load_torch(VoiceLM, MODEL_PATH)

audio, sr = load(
    "src/experiments/data/sir.wav",
    normalize=True,
    channels_first=True,
    backend="ffmpeg",
)
audio = audio[0].numpy()

model.to("cuda")
tokens = model.generate(
    ["How are you my dear?", audio],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
)
output = model.tokenizer.decode(tokens[0].tolist())
print(output)
