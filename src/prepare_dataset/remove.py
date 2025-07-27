import asyncio
from pathlib import Path


async def remove_tars(tar_files: list[Path]) -> None:
    tasks = []
    for tar_file in tar_files:
        task = asyncio.create_task(asyncio.to_thread(tar_file.unlink, missing_ok=True))
        tasks.append(task)

    await asyncio.gather(*tasks)
    print("Done")
