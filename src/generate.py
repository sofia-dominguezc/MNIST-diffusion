import random
from tqdm import tqdm
from typing import Optional, Callable, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from architectures import Diffusion, AutoEncoder, VarAutoEncoder
from ml_utils import plot_images


def process_labels(
    labels: Sequence[Optional[int]],
    n_classes: int,
    device: str | torch.device,
) -> Tensor:
    """
    Converts labels to one-hot and handles None
    out: (len(labels), n_classes)
    """
    no_nan_labels = [l if l is not None else n_classes for l in labels]
    y = torch.tensor(no_nan_labels, device=device)
    y = F.one_hot(y, num_classes=n_classes + 1).to(torch.float32)
    return y[..., :-1]  # delete None label


class SDESolver:
    """
    Class that implements an SDE Solver for integrating the nn model
    Uses the Euler Maruyama method because it does less NN calls
    """
    def __init__(
        self,
        model: Diffusion,
        y: Optional[Tensor] = None,
        weight: float = 1,
        diffusion: float | Callable[[Tensor], Tensor] = 1,
    ):
        self.model = model
        self.y = y
        self.weight = weight
        self.diffusion = diffusion
        self.device = next(model.parameters()).device
        self.n_classes = model.n_classes

    def noise_fn(self, t: Tensor) -> Tensor:
        """Function that determines the level of noise at each time"""
        if isinstance(self.diffusion, (float, int)):
            return self.diffusion * (1 - t)
        else:
            return self.diffusion(t) * (1 - t)

    def _flow(self, x: Tensor, t: Tensor) -> Tensor:
        """Wrapper of model.forward that handles the weight and the no label case"""
        if self.y is None:
            no_y = torch.zeros((x.shape[0], self.model.n_classes), device=self.device)
            return self.model(x, t, no_y)
        else:
            no_y = torch.zeros_like(self.y)
            conditioned = self.model(x, t, self.y)
            unconditioned = self.model(x, t, no_y)
            return self.weight * conditioned + (1 - self.weight) * unconditioned

    def _score(self, flow: Tensor, x: Tensor, t: Tensor):
        """
        Calculates score in a numerically stable way
        (using noise_fn directly would lead to division by 0 at t=1)
        """
        if isinstance(self.diffusion, (float, int)):
            return self.diffusion * (t * flow - x)
        else:
            return self.diffusion(t) * (t * flow - x)

    def f(self, x: Tensor, t: Tensor) -> Tensor:
        flow = self._flow(x, t)
        return flow + 0.5 * self.noise_fn(t)**2 * self._score(flow, x, t)

    def g(self, x: Tensor, t: Tensor) -> Tensor:
        return self.noise_fn(t)

    def integrate(
        self, x0: Tensor, t0: float = 0, t1: float = 1, num_steps: int = 50, 
    ) -> Tensor:
        """Numerically integrate dx = f(x, t)dx + g(x, t)dW"""
        h = (t1 - t0) / num_steps
        t_shape = [x0.shape[0]] + [1] * (len(x0.shape) - 1)
        t = t0 + torch.zeros(t_shape, device=x0.device)
        x = x0
        for _ in tqdm(range(num_steps), desc="Integrating SDE"):
            ep = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            x = x + self.f(x, t) * h + self.g(x, t) * h**0.5 * ep
            t += h
        return x


def diffusion_generation(
    model: Diffusion,
    autoencoder: AutoEncoder | VarAutoEncoder,
    labels: Optional[Sequence[Optional[int]]] = None,
    weight: float = 1,
    diffusion: float | Callable[[Tensor], Tensor] = 1.0,
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
    device = next(model.parameters()).device
    if labels is None:
        y = None
    else:
        y = process_labels(labels, model.n_classes, device=device)
    solver = SDESolver(model, y=y, weight=weight, diffusion=diffusion)

    z0 = torch.normal(
        torch.zeros((n_imgs, *autoencoder.z_shape), device=solver.device, dtype=torch.float32),
        torch.ones((n_imgs, *autoencoder.z_shape), device=solver.device, dtype=torch.float32),
    )

    with torch.no_grad():
        z1 = solver.integrate(z0, t0=0, t1=1, num_steps=num_steps)
        x1 = autoencoder.decode(z1)
        imgs = torch.clip(x1.detach().cpu(), 0, 1)  # (n_imgs, 1, 28, 28)

    plot_images(
        imgs.view(height, width, *imgs.shape[2:]),
        figsize=(scale*width, scale*height),
    )


def autoencoder_reconstruction(
    autoencoder: AutoEncoder | VarAutoEncoder,
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
    device = next(autoencoder.parameters()).device
    num_batches = len(dataloader)

    batch_idx = random.choice(range(1, num_batches - 1))
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= batch_idx:
                break

    imgs = x.to(device)  # type: ignore
    assert imgs.shape[0] == height * width, "batch_size must be set to height * width"
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


def classify(
    model: Diffusion,
    autoencoder: AutoEncoder | VarAutoEncoder,
    x: Tensor,
    weight: float = 1,
    num_steps: int = 100,
) -> Tensor:
    """
    Classify x using the flow model and autoencoder
    Uses Bayes rule: p(y | x) ~ sum_y p(x | y) p(y)
    NOTE: doesn't work for splits other than "balanced" in EMNIST
    """
    n_classes = model.n_classes
    batches = x.shape[0]
    z = autoencoder.encode(x)
    z1 = z.unsqueeze(0).repeat(n_classes, *([1] * z.ndim))  # (n_class, batch, *z_shape)

    labels = process_labels(list(range(n_classes)), n_classes, z.device)  # (n_class, n_class)
    y = labels.unsqueeze(1).repeat(1, batches, 1)  # (n_class, batch, n_class)

    solver = SDESolver(model, y.flatten(0, 1), weight=weight, diffusion=0)
    z0_flat = solver.integrate(z1.flatten(0, 1), t0=1, t1=0, num_steps=num_steps)
    z0 = z0_flat.unflatten(0, (n_classes, batches)).flatten(2)  # (n_class, batches, prod(z_shape))

    probs = torch.exp(- 0.5 * torch.sum(z0**2, dim=-1))  # (n_class, batches)
    probs = probs / torch.sum(probs, dim=0, keepdim=True)
    return probs.T  # (batches, n_class)
