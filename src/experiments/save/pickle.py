import shutil
from pathlib import Path

from src.constants.train import PRESETS_FACTORY
from src.utils.model_io import save_pickle, load_pickle
from src.models.voicelm import VoiceLM

SAVE_DIR = Path("./.temp").resolve()
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model = PRESETS_FACTORY["asr/whisper"]()
print(model.generate(["Hello world!"]))

save_pickle(model, SAVE_DIR / "model.pkl")

model = load_pickle(VoiceLM, SAVE_DIR / "model.pkl")
print(model.generate(["Hello world!"]))
