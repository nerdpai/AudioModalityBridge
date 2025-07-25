import asyncio
from pathlib import Path

import aiohttp


async def _download_file(url: str, destination: Path) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            with destination.open("wb") as file:
                async for chunk in response.content.iter_chunked(8192):
                    file.write(chunk)


async def download_file(url: str, destination: Path) -> None:
    if not url.startswith("http://") or not url.startswith("https://"):
        url = f"https://{url}"

    await _download_file(url, destination)
    print(f"Download from {url} ended.")


async def download_dataset(files_to_download: list[tuple[Path, Path]]) -> None:
    promises = []
    for url, destination in files_to_download:
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            print(f"File {destination} already exists, skipping download.")
            continue

        promise = download_file(str(url), destination)
        promises.append(promise)

    await asyncio.gather(*promises)
