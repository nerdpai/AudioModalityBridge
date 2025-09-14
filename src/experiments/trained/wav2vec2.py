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
tokenizer = model.tokenizer

model.to("cuda")

template = [
    {
        "role": "system",
        "content": "You are an assistant that continuous the user's input.",
    },
    {
        "role": "user",
        "content": "",
    },
]
instruction: str = tokenizer.apply_chat_template(template, tokenize=False)  # type: ignore
eos_token: str = tokenizer.eos_token  # type: ignore
instruction = instruction.removesuffix(eos_token)

tokens = model.generate(
    [[instruction, audio, eos_token]],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
)
output = model.tokenizer.decode(tokens[0].tolist())
print(output)
