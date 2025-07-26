import re
import asyncio
from pathlib import Path

from .constants import BASE_URL, LOCALE, LIMITS, CACHE_DIR, DATASET_DIR
from .load import download_dataset
from .dearchivate import dearchivate_dataset
from .filter import filter_dataset
from .move import move_dataset


def title_print(message: str) -> None:
    prefix_matches = re.match(r"^(\n*).*", message)
    sufix_matches = re.match(r"^(\n*).*", message[::-1])

    prefix = prefix_matches.group(1)  # type: ignore
    sufix = sufix_matches.group(1)  # type: ignore

    print(f"{prefix}{'-' * 10}{message.strip()}{'-' * 10}{sufix}")


def prepare_files_to_download() -> list[tuple[Path, Path]]:
    files_to_download = []

    for split in LIMITS.keys():
        file_path = f"transcript/{LOCALE}/{split}.tsv"
        files_to_download.append((BASE_URL / file_path, CACHE_DIR / file_path))

    for split, limit in LIMITS.items():
        shard_path = f"audio/{LOCALE}/{split}"
        for shard in range(limit.NUMBER_OF_SHARDS):
            file_path = f"{shard_path}/{LOCALE}_{split}_{shard}.tar"
            files_to_download.append((BASE_URL / file_path, CACHE_DIR / file_path))

    return files_to_download


def run():
    files_to_download = prepare_files_to_download()

    title_print("Fetching data\n")
    asyncio.run(download_dataset(files_to_download))

    title_print("\n\nDearchivating data\n")
    asyncio.run(
        dearchivate_dataset([f[1] for f in files_to_download if f[1].suffix == ".tar"])
    )

    title_print("\n\nFilter data\n")
    asyncio.run(
        filter_dataset(
            [f[1] for f in files_to_download if f[1].suffix == ".tsv"],
            CACHE_DIR / f"audio/{LOCALE}",
        )
    )

    title_print("\n\nMove files to dataset dir")
    move_dataset(CACHE_DIR, DATASET_DIR)
