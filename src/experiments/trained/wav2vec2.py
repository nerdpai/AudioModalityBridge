from pathlib import Path

from torchaudio import load

from src.constants.train import PRESETS_FACTORY
from src.utils.model_io import load_torch
from src.models.voicelm import VoiceLM

MODEL_PATH = Path(".models/classification/wav2vec2/model.pt").resolve()
model = load_torch(
    VoiceLM, lambda: PRESETS_FACTORY["classification/wav2vec2"](), MODEL_PATH
)

audio, sr = load(
    "src/experiments/data/sir.wav",
    normalize=True,
    channels_first=True,
    backend="ffmpeg",
)
audio = audio[0].numpy()

model.to("cuda")
tokens = model.generate(
    [["Tell me about yourself."]],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
)
output = model.tokenizer.decode(tokens[0].tolist())
print(output)
