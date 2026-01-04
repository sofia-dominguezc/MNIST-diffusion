import random
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Callable, Sequence, Literal

import torch
from torch import Tensor

from diffusion.architecture import Diffusion, DiffusionCNN, DiffusionViT
from utils import plot_images, sample_noise, load_model, get_num_classes, parse_unknown_args
import os
from PIL import Image
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "EMNIST", "FashionMNIST"], required=True)
    parser.add_argument("--split", choices=["balanced", "byclass", "bymerge"], default="balanced")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--fid", action="store_true", help="If true, also computes FID metric")
    parser.add_argument("--batch-size", type=int, default=1024, help="For FID computation")

    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--scale", type=float, default=0.8)

    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--model-version", choices=["dev", "main"], default="main")
    parser.add_argument("--noise-type", choices=["gaussian", "uniform", "bernoulli"], default="gaussian")

    parser.add_argument("--weight", type=float, default=2.0, help="Classifier-free guidance weight")
    parser.add_argument("--cfg-start", type=float, default=0.3)
    parser.add_argument("--cfg-end", type=float, default=0.7)
    parser.add_argument("--diffusion", type=float, default=0.0, help="level of noise")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of steps in integration")

    args, unknown = parser.parse_known_args()
    return args, unknown


def process_labels(
    labels: Sequence[int | None],
    device: str | torch.device,
) -> Tensor:
    """
    Converts labels to one-hot and handles None
    out: (len(labels), n_classes)
    """
    no_nan_labels = [l if l is not None else -1 for l in labels]
    return torch.tensor(no_nan_labels, device=device, dtype=torch.long)


class SDESolver:
    """
    Class that implements an SDE Solver for integrating the nn model
    Uses the Euler Maruyama method because it does less NN calls
    """
    def __init__(
        self,
        model: Diffusion,
        y: Tensor | None = None,
        weight: float = 1,
        cfg_start: float = 0.3,
        cfg_end: float = 0.7,
        diffusion: float | Callable[[Tensor], Tensor] = 1,
    ):
        self.model = model
        self.y = y

        self.weight = weight
        self.cfg_start = cfg_start
        self.cfg_end = cfg_end
        self.diffusion = diffusion

        self.device = next(model.parameters()).device

    def noise_fn(self, t: Tensor) -> Tensor:
        """Function that determines the level of noise at each time"""
        if isinstance(self.diffusion, (float, int)):
            return self.diffusion * (1 - t)
        else:
            return self.diffusion(t) * (1 - t)

    def _flow(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Return v = (x_pred - xt) / (1 - t) with CFG
        Use -1 for the no label embedding    
        """
        if self.y is None:
            no_y = - torch.ones(x.shape[0], device=self.device, dtype=x.dtype)
            return (self.model(x, t, no_y) - x) / (1 - t)
        else:
            no_y = - torch.ones_like(self.y)
            conditioned = (self.model(x, t, self.y) - x) / (1 - t)
            unconditioned = (self.model(x, t, no_y) - x) / (1 - t)
            weight = torch.where((self.cfg_start < t) & (t < self.cfg_end), self.weight, 1.0)
            return weight * conditioned + (1 - weight) * unconditioned

    def _score(self, flow: Tensor, x: Tensor, t: Tensor):
        """
        Calculates score in a numerically stable way
        (using noise_fn directly would lead to division by 0 at t=1)
        """
        if isinstance(self.diffusion, (float, int)):
            return self.diffusion * (t * flow - x)
        else:
            return self.diffusion(t) * (t * flow - x)

    def f(self, x: Tensor, t: Tensor, sign: Literal[1, -1] = 1) -> Tensor:
        flow = self._flow(x, t)
        return flow + sign * 0.5 * self.noise_fn(t)**2 * self._score(flow, x, t)

    def g(self, x: Tensor, t: Tensor) -> Tensor:
        return self.noise_fn(t)

    def solve(
        self,
        x0: Tensor,
        t0: float = 0,
        t1: float = 1,
        num_steps: int = 100,
    ) -> Tensor:
        """
        Numerically integrate dx = f(x, t)dx + g(x, t)dW
        Supports both forward and backward integration
        """
        sign = 1 if t1 > t0 else -1
        T = torch.linspace(t0, t1, num_steps + 1, device=x0.device)
        dT = T[1:] - T[:-1]

        x = x0
        pbar = tqdm(desc="Integrating SDE", total=num_steps, disable=x.device!=torch.device("cpu"))
        for _t, dt in zip(T, dT):
            t = _t.repeat([x.shape[0]] + [1] * (x.ndim - 1))
            ep = torch.normal(torch.zeros_like(x), torch.ones_like(x))
            x = x + self.f(x, t, sign) * dt + self.g(x, t) * dt.abs()**0.5 * ep
            pbar.update(1)
        pbar.close()
        return x


def diffusion_generation(
    model: Diffusion,
    autoencoder=None,
    labels: Sequence[int | None] | None = None,
    weight: float = 1,
    cfg_start: float = 0.3,
    cfg_end: float = 0.7,
    diffusion: float | Callable[[Tensor], Tensor] = 1.0,
    width: int = 10,
    height: int = 10,
    scale: float = 1,
    num_steps: int = 25,
    noise_type: Literal["gaussian", "uniform", "bernoulli"] = "gaussian",
    plot=True,
):
    """
    Samples random noise and uses flow model to recover images
    (which may be in an autoencoder's latent space)

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
    z_shape = autoencoder.z_shape if autoencoder is not None else (1, 28, 28)
    device = next(model.parameters()).device
    y = None if labels is None else process_labels(labels, device=device)
    solver = SDESolver(
        model, y=y, weight=weight, cfg_start=cfg_start, cfg_end=cfg_end, diffusion=diffusion,
    )

    z0 = sample_noise(noise_type, (n_imgs, *z_shape), solver.device, torch.float32)
    with torch.no_grad():
        z1 = solver.solve(z0, t0=0, t1=1, num_steps=num_steps)
        x1 = autoencoder.decode(z1) if autoencoder is not None else z1
        imgs = x1.detach().cpu().clip(0, 1)  # (n_imgs, 1, 28, 28)

    if plot:
        plot_images(
            imgs.view(height, width, *imgs.shape[2:]),
            figsize=(scale*width, scale*height),
        )
    return imgs


def sample(args, plot=True, **nn_kwargs):
    arch = DiffusionCNN if args.model == "cnn" else DiffusionViT
    n_classes = get_num_classes(args.dataset, args.split)
    model = load_model(
        model_architecture=arch, dataset=args.dataset,
        model_version=args.model_version, **nn_kwargs,
        n_classes=n_classes,
    ).to(args.device)

    imgs = diffusion_generation(  # (n_imgs, 1, 28, 28)
        model,
        autoencoder=None,
        labels=[k % n_classes for k in range(args.height * args.width)],
        weight=args.weight,
        diffusion=args.diffusion,
        width=args.width,
        height=args.height,
        scale=args.scale,
        num_steps=args.num_steps,
        noise_type=args.noise_type,
        plot=plot,
    )
    return imgs


def save_generated_images(
    args,
    output_dir: str = "data/generated",
    batch_size: int = 256,
    n_images: int = 50000,
    **nn_kwargs,
):
    """
    Generate and save N_images as individual PNG files.

    Args:
        args: parsed arguments
        output_dir: directory to save images
        batch_size: number of images to generate per batch
        n_images: total number of images to generate
        **nn_kwargs: additional model kwargs
    """
    os.makedirs(output_dir, exist_ok=True)

    arch = DiffusionCNN if args.model == "cnn" else DiffusionViT
    n_classes = get_num_classes(args.dataset, args.split)
    model = load_model(
        model_architecture=arch, dataset=args.dataset,
        model_version=args.model_version, **nn_kwargs,
        n_classes=n_classes,
    ).to(args.device)

    num_batches = n_images // batch_size
    image_count = 0

    for _ in tqdm(range(num_batches), desc="Generating and saving images"):
        current_batch_size = min(batch_size, n_images - image_count)

        imgs = diffusion_generation(
            model,
            autoencoder=None,
            labels=[random.randint(0, 10) for k in range(current_batch_size)],
            weight=args.weight,
            diffusion=args.diffusion,
            width=int(np.sqrt(current_batch_size)),
            height=int(np.ceil(current_batch_size / np.sqrt(current_batch_size))),
            scale=args.scale,
            num_steps=args.num_steps,
            noise_type=args.noise_type,
            plot=False,
        )

        for i in range(current_batch_size):
            img_array = (imgs[i].squeeze().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            img.save(os.path.join(output_dir, f"image_{image_count:06d}.png"))
            image_count += 1


if __name__ == "__main__":
    args, unknown = parse_args()
    nn_kwargs = parse_unknown_args(unknown)
    sample(args, plot=True, **nn_kwargs)
    if args.fid:
        save_generated_images(
            args, output_dir="data/generated",
            batch_size=args.batch_size, n_images=60000, **nn_kwargs,
        )
