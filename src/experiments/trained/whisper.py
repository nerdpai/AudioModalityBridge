from pathlib import Path

from torchaudio import load

from src.constants.train import PRESETS_FACTORY
from src.utils.model_io import load_torch
from src.models.voicelm import VoiceLM

MODEL_PATH = Path(".models/asr/whisper/model.pt").resolve()
model = load_torch(VoiceLM, None, MODEL_PATH)

audio, sr = load(
    ".datasets/mozilla_common_voice/audio/en/test/common_voice_en_1075.mp3",
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
        "content": "You are a helpful assistant who continues the user's input",
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
    [[instruction, audio, eos_token, "assistant:"]],
    max_new_tokens=100,
    do_sample=True,
    temperature=1.0,
)
output = model.tokenizer.decode(tokens[0].tolist())
print(output)
