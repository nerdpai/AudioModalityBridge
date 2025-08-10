import numpy as np
from torch.utils.data import DataLoader

from .dataset import CommonVoice, Splits


def collate_fn(
    batch: list[tuple[np.ndarray, str]],
) -> tuple[list[np.ndarray], list[str]]:
    audio_data, transcripts = zip(*batch)
    return list(audio_data), list(transcripts)


def get_loader(
    split: Splits, batch_size: int, num_workers: int
) -> DataLoader[tuple[np.ndarray, str]]:
    dataset = CommonVoice.from_constants(split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
