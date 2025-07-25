import asyncio
import tarfile
import shutil
from pathlib import Path


async def _dearchivate_shard(tar_file: Path) -> None:
    await asyncio.to_thread(extract_tar, tar_file, tar_file.parent)

    files_dir = tar_file.parent / tar_file.stem
    await move_files(files_dir, tar_file.parent)

    files_dir.rmdir()


async def dearchivate_shard(tar_file: Path) -> None:
    if not tar_file.exists():
        raise FileNotFoundError(f"Tar file {tar_file} does not exist.")

    await _dearchivate_shard(tar_file)
    print(f"Shard {tar_file} dearchived.")


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_to)


async def move_files(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tasks = []

    for item in source_dir.glob("*"):
        if item.is_file():
            task = asyncio.create_task(
                asyncio.to_thread(shutil.move, item, dest_dir / item.name)
            )
            tasks.append(task)

    await asyncio.gather(*tasks)


async def dearchivate_dataset(files_to_dearchivate: list[Path]) -> None:
    tasks = [dearchivate_shard(tar_file) for tar_file in files_to_dearchivate]
    await asyncio.gather(*tasks)
