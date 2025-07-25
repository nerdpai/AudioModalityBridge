from pathlib import Path
import shutil


def move_dataset(dir_src: Path, dir_dst: Path) -> None:
    if dir_dst.exists():
        shutil.rmtree(dir_dst, ignore_errors=True)
    shutil.move(dir_src, dir_dst)
    shutil.rmtree(dir_src, ignore_errors=True)
