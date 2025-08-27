import random
from typing import Optional, Callable, Union, cast

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchsde

from architectures import Diffusion, AutoEncoder, VarAutoEncoder
from ml_utils import dummy_noise, plot_images


def flow_fn(
    model: Diffusion,
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
    model: Diffusion,
    labels: list[Optional[int]],
) -> Tensor:
    """Converts y to one-hot and handles Nones"""
    labels = [l if l is not None else model.n_classes for l in labels]
    y = torch.tensor(labels, device=next(model.parameters()).device)
    y = F.one_hot(y, num_classes=model.n_classes + 1).to(torch.float32)
    return y[..., :-1]  # delete None label


class SDESolver(nn.Module):
    """Class that implements an SDE Solver for integrating the nn model"""
    def __init__(
        self,
        model: Diffusion,
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
        return self.noise_fn(t)


def integrate_flow(
    model: Diffusion,
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


def diffusion_generation(
    model: Diffusion,
    autoencoder: Union[AutoEncoder, VarAutoEncoder],
    labels: Optional[list[Optional[int]]] = None,
    w: float = 1,
    noise_fn: Callable[[torch.Tensor], torch.Tensor] = dummy_noise,
    width: int = 10,
    height: int = 10,
    scale: float = 1,
    num_steps: int = 50,
):
    """
    Samples random noise, then uses the flow model to carry them to
    the latent space of the autoencoder, then recovers the images.

    Args:
        width: number of images to have horizontally
        height: number of images to produce vertically
        model: diffusion model. Space must match with input of decoder
        autoencoder: autoencoder
        labels: if provided, list of length width*height with labels.
            None refers to no labels for that particular image.
        w: weight of the condition for flow.
        noise_fn: diffussion coefficient.
        scale: size of each image to plot.
        num_steps: number of steps in integration
    """
    n_imgs = height * width
    z_shape = autoencoder.z_shape
    device = next(model.parameters()).device

    ones = torch.empty((n_imgs, *z_shape), device=device, dtype=torch.float32)
    z0 = torch.normal(torch.zeros_like(ones), torch.ones_like(ones))

    with torch.no_grad():
        z1 = integrate_flow(
            model, z0, labels, w, noise_fn, num_steps, t0=0, t1=1, full_output=False
        )
        x1 = autoencoder.decode(z1)
        imgs = torch.clip(x1.detach().cpu(), 0, 1)  # (n_imgs, 1, 28, 28)

    plot_images(
        imgs.view(height, width, *imgs.shape[2:]),
        figsize=(scale*width, scale*height),
    )


def autoencoder_reconstruction(
    autoencoder: Union[AutoEncoder, VarAutoEncoder],
    dataloader: DataLoader,
    width: int = 10,
    height: int = 10,
    scale: int = 1,
):
    """
    Randomly sample image from dataloader and plot it
    together with the autoencoder's reconstruction

    Args:
        autoencoder: model to use
        dataloader: where to get images from
        width: how many images horizontally (along with reconstructions)
        height: how many images to plot vertically
    """
    num_img = width * height
    device = next(autoencoder.parameters()).device
    num_batches = len(dataloader)

    batch_idx = random.choice(range(num_batches))
    x, y = dataloader[batch_idx]  # type: ignore

    list_imgs = random.choices(x, k=num_img)
    imgs = torch.stack(list_imgs).to(device)
    pred_imgs = autoencoder.decode(autoencoder.encode(imgs))

    imgs = imgs[:, 0].cpu().detach()  # (num_img, 28, 28)
    imgs = imgs.view(height, width, *imgs.shape[-2:])
    pred_imgs = pred_imgs[:, 0].cpu().detach()
    pred_imgs = torch.clip(pred_imgs, 0, 1)
    pred_imgs = pred_imgs.view(height, width, *pred_imgs.shape[-2:])

    combined_imgs = torch.empty((height, 2*width, *imgs.shape[2:]))
    combined_imgs[:, ::2] = imgs
    combined_imgs[:, 1::2] = pred_imgs

    plot_images(
        combined_imgs,
        figsize=(2*scale*width, scale*height),
    )
