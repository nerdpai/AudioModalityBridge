import shutil
from pathlib import Path

from src.constants.train import PRESETS_FACTORY
from src.utils.model_io import save_torch, load_torch
from src.models.voicelm import VoiceLM

SAVE_DIR = Path("./.temp").resolve()
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model = PRESETS_FACTORY["asr/whisper"]()
print(model.generate(["Hello world!"]))

save_torch(model, SAVE_DIR / "model.pt")

model = load_torch(VoiceLM, SAVE_DIR / "model.pt")
print(model.generate(["Hello world!"]))
