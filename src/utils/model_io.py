from pathlib import Path
from typing import Any, TypeVar, Type
import pickle

import torch
import torch.nn as nn

T = TypeVar("T")


def save_torch(
    model: nn.Module,
    filepath: Path,
) -> None:
    save_data = {
        "model": model,
        "model_state": model.state_dict(),
    }

    torch.save(save_data, filepath)


def load_torch(model_t: Type[T], filepath: Path) -> T:
    checkpoint = torch.load(filepath, map_location="cpu")
    return checkpoint["model"]


def save_pickle(obj: Any, filepath: Path) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(obj_t: Type[T], filepath: Path) -> T:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_checkpoint(model: nn.Module, filepath: Path, **kwargs: Any) -> None:
    save_data = {"model_state": model.state_dict(), **kwargs}
    torch.save(save_data, filepath)


def load_checkpoint(filepath: Path) -> dict[str, Any]:
    return torch.load(filepath, map_location="cpu")
