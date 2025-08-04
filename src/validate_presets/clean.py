import torch
from torch import nn


def cleanup_model(model: nn.Module):
    model.zero_grad()
    model.cpu()
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
