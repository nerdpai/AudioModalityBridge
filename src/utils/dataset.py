from pathlib import Path

import numpy as np
import pandas as pd
from pandera.typing.pandas import DataFrame
from torch.utils.data import Dataset
from torchaudio import load
import torchaudio.functional as AF

from src.types.dataset import CommonVoiceModel, Splits
from src.constants.dataset import SAMPLING_RATE, DATASET_DIR, LOCALE


class CommonVoice(Dataset):
    def __init__(
        self,
        tsv_file: Path,
        audio_dir: Path,
        sr: int,
    ):
        self.tsv_file: DataFrame[CommonVoiceModel] = pd.read_csv(tsv_file, sep="\t")  # type: ignore
        self.audio_dir = audio_dir
        self.desired_sr = sr

    def __len__(self):
        return self.tsv_file.shape[0]

    def __getitem__(self, idx) -> tuple[np.ndarray, str]:
        row = self.tsv_file.iloc[idx]
        audio_path = self.audio_dir / row.path

        audio, sr = load(
            audio_path, normalize=True, channels_first=True, backend="ffmpeg"
        )
        audio = AF.resample(audio, sr, self.desired_sr)
        audio = audio.squeeze(0)

        return audio.numpy(), row.sentence

    @classmethod
    def from_constants(cls, split: Splits) -> "CommonVoice":
        tsv_file = DATASET_DIR / "transcript" / LOCALE / f"{split}.tsv"
        audio_dir = DATASET_DIR / "audio" / LOCALE / split

        if not tsv_file.exists():
            raise FileNotFoundError(f"TSV file not exist: {tsv_file}")

        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not exist: {audio_dir}")

        return cls(tsv_file, audio_dir, SAMPLING_RATE)
