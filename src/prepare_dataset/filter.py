import asyncio
import shutil
from pathlib import Path
from typing import Awaitable


import pandas as pd
from pandera.typing.pandas import DataFrame

from .types import CommonVoiceModel
from .constants import LIMITS

BATCH_SIZE = int(1e3)


def process_batch(
    df: DataFrame[CommonVoiceModel],
    data_dir: Path,
    up_votes: int,
    down_votes: int,
):
    rm_files = []

    for row in list(df.itertuples()):
        audio_path = data_dir / row.path  # type: ignore

        if not audio_path.exists():
            df.drop(row.Index, inplace=True)
            continue

        if row.up_votes < up_votes or row.down_votes > down_votes:  # type: ignore
            df.drop(row.Index, inplace=True)
            rm_files.append(audio_path)

    return df, rm_files


async def _filter_path(
    df: DataFrame[CommonVoiceModel], data_dir: Path
) -> DataFrame[CommonVoiceModel]:
    rm_dir = data_dir / "rm"
    rm_dir.mkdir(parents=True, exist_ok=True)

    split = data_dir.name
    up_votes = LIMITS[split].UP_VOTES  # type: ignore
    down_votes = LIMITS[split].DOWN_VOTES  # type: ignore

    tasks = []
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE].copy()
        task = asyncio.create_task(
            asyncio.to_thread(process_batch, batch, data_dir, up_votes, down_votes)
        )
        tasks.append(task)

    dfs = []
    rm_files = []
    for df, rm_f in await asyncio.gather(*tasks):
        dfs.append(df)
        rm_files.extend(rm_f)

    df = pd.concat(dfs, ignore_index=True)  # type: ignore

    await move_files(rm_files, rm_dir)
    shutil.rmtree(rm_dir)

    return df


async def filter_path(
    df: DataFrame[CommonVoiceModel], data_dir: Path
) -> DataFrame[CommonVoiceModel]:
    if data_dir.exists():
        filtered = await _filter_path(df, data_dir)
    else:
        print(f"Filter path {data_dir} does not exist, skipping...")
        filtered = df

    print(f"Filtered for {data_dir.name}.tsv.")
    return filtered


async def move_files(files: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tasks = []

    for file in files:
        task = asyncio.create_task(
            asyncio.to_thread(shutil.move, file, dest_dir / file.name)
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


def read_tsv(file_path: Path) -> DataFrame[CommonVoiceModel]:
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    return pd.read_csv(file_path, sep="\t")  # type: ignore


async def filter_dataset(filter_files: list[Path], data_dir: Path) -> None:
    promises: list[Awaitable[DataFrame[CommonVoiceModel]]] = []
    for file_path in filter_files:
        df = read_tsv(file_path)
        filtered_promise = filter_path(df, data_dir / file_path.stem)
        promises.append(filtered_promise)

    filtered_dfs = await asyncio.gather(*promises)
    for file_path, filtered_df in zip(filter_files, filtered_dfs):
        if file_path.exists():
            file_path.unlink()
        filtered_df.to_csv(file_path, sep="\t", index=False)
