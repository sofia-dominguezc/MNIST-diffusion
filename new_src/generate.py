from typing import Optional, Callable
from math import prod

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchsde

import new_src.train as models


def flow_fn(
    model: models.Diffusion,
    xt: Tensor,
    t: Tensor,
    y: Optional[Tensor] = None,
    w: float = 1,
) -> Tensor:
    if y is None:
        no_y = torch.zeros((xt.shape[0], model.n_classes), device=xt.device)
        return model(xt, t, no_y)
    else:
        no_y = torch.zeros_like(y)
        return w * model(xt, t, y) + (1 - w) * model(xt, t, no_y)


def score_fn(
    model: models.Diffusion,
    xt: Tensor,
    t: Tensor,
    y: Optional[Tensor] = None,
    w: float = 1,
) -> Tensor:
    return (t * flow_fn(model, xt, t, y, w=w) - xt)/(1 - t)


def process_labels(
    model: models.Diffusion,
    labels: list[Optional[int]],
) -> Tensor:
    """Converts y to one-hot and handles Nones"""
    labels = [l if l is not None else model.n_classes for l in labels]
    y = torch.tensor(labels, device=model.device)
    y = F.one_hot(y, num_classes=model.n_classes + 1).to(torch.float32)
    return y[..., :-1]  # delete None label


def dummy_noise(t: Tensor) -> Tensor:
    return Tensor(1.0)


class SDESolver(nn.Module):
    """Class that implements an SDE Solver for generating images"""
    def __init__(
        self,
        model: models.Diffusion,
        labels: Optional[list[Optional[int]]] = None,
        w: float = 1,
        noise_fn: Callable[[Tensor], Tensor] = dummy_noise,
    ):
        super().__init__()
        self.model = model
        self.w = w
        self.noise_fn = noise_fn
        if labels is None:
            self.y = None
        else:
            self.y = process_labels(model, labels)

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        drift = flow_fn(self.model, x, t, y=self.y, w=self.w)
        diffusion = score_fn(self.model, x, t, y=self.y, w=self.w)
        return drift + 0.5 * self.noise_fn(t)**2 * diffusion

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return self.noise_fn(t).unsqueeze(-1)
