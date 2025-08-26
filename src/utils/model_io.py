from pathlib import Path
from typing import Any, TypeVar, Type, Callable, Optional
import pickle

import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


def save_torch(
    model: nn.Module,
    filepath: Path,
) -> None:
    try:
        torch.save({"model": model}, filepath)
    except Exception as e:
        if filepath.exists():
            filepath.unlink()

        print(f"Could not save complete model: {e}")
        print("Falling back to state_dict")

        torch.save({"model_state": model.state_dict()}, filepath)


def load_torch(
    model_t: Type[T],
    initializer: Optional[Callable[[], T]],
    filepath: Path,
) -> T:
    checkpoint = torch.load(
        filepath,
        map_location="cpu",
        weights_only=initializer is not None,
    )

    if initializer is None:
        return checkpoint["model"]
    else:
        model = initializer()
        model.load_state_dict(checkpoint["model_state"])
        return model


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
