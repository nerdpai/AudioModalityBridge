import torch
from torchaudio import load
from transformers.models.wav2vec2 import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

from src.constants.secrets import SECRETS


MODEL_ID = "somosnlp-hackathon-2022/wav2vec2-base-finetuned-sentiment-mesd"

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="9bc54869d8082fc35adcbbf6f603e8365621f8e0",
    torch_dtype=torch.float32,
    device_map="cuda",
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_ID,
    token=SECRETS["HF_ACCESS_TOKEN"],
    revision="9bc54869d8082fc35adcbbf6f603e8365621f8e0",
)

audio, sr = load(
    "src/experiments/data/war_face.wav",
    normalize=True,
    channels_first=True,
    backend="ffmpeg",
)
audio = audio[0].numpy()
print(f"Audio shape: {audio.shape}")
input_values = feature_extractor(
    audio, return_tensors="pt", sampling_rate=16000
).input_values

input_values = input_values.to(dtype=torch.float32)

output = model(input_values.to(model.device)).logits
output = output.argmax(dim=-1).item()

output = model.config.id2label[output]

print(output)  # happy war face XD
