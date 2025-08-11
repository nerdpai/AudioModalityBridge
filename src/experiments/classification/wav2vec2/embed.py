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
model = model.wav2vec2

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
features = feature_extractor(
    audio, return_tensors="pt", sampling_rate=16000, return_attention_mask=True
)

input_values = features.input_values
attention_mask: torch.Tensor = features.attention_mask
print(f"Attention mask tensor type: {type(attention_mask)}")
print(f"Attentions mask values: {attention_mask[:5]}")

input_values = input_values.to(dtype=torch.float32)
# attention_mask = torch.zeros_like(attention_mask, dtype=torch.int32)

with torch.no_grad():
    output = model.forward(
        input_values.to(model.device), attention_mask.to(model.device)
    )

print(output.last_hidden_state.shape)
print(output.last_hidden_state[0, -5:-1, :5])
