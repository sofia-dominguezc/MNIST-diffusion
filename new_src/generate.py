from typing import Optional, Callable, cast

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchsde

import new_src.train as plmodels


def flow_fn(
    model: plmodels.Diffusion,
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


def score_fn(flow_output: Tensor, xt: Tensor, t: Tensor) -> Tensor:
    return (t * flow_output - xt)/(1 - t)


def process_labels(
    model: plmodels.Diffusion,
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
        model: plmodels.Diffusion,
        y: Optional[Tensor] = None,
        w: float = 1,
        noise_fn: Callable[[Tensor], Tensor] = dummy_noise,
    ):
        super().__init__()
        self.model = model
        self.y = y
        self.w = w
        self.noise_fn = noise_fn

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        drift = flow_fn(self.model, x, t, y=self.y, w=self.w)
        return drift + 0.5 * self.noise_fn(t)**2 * score_fn(drift, x, t)

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return self.noise_fn(t).unsqueeze(-1)  # add brownian_dim


def integrate_flow(
    model: plmodels.Diffusion,
    z0: Tensor,
    labels: Optional[list[Optional[int]]] = None,
    w: float = 1,
    noise_fn: Callable[[Tensor], Tensor] = dummy_noise,
    num_steps: int = 50,
    t0: float = 0,
    t1: float = 1,
    full_output: bool = False
) -> Tensor:
    """
    Integrates the SDE from t0 to t1 to get z1 from z0
    If provided, labels must be a list of length num_batches
    """
    if labels is None:
        y = None
    else:
        assert len(labels) == z0.shape[0], "Incorrect labels length"
        y = process_labels(model, labels)
    solver = SDESolver(model, y=y, w=w, noise_fn=noise_fn)
    ts = torch.linspace(t0, t1, num_steps)
    z = torchsde.sdeint(solver, z0, ts)
    if full_output:
        return cast(Tensor, z)
    return z[-1]
