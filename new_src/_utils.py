from typing import Optional, Callable, Union, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from new_src.train import GeneralModel, Diffusion, AutoEncoder, VarAutoEncoder
from new_src.generate import dummy_noise, integrate_flow


def train(
    pl_class: type[GeneralModel],
    train_loader: DataLoader,
    lr: float,
    total_epochs: int,
    milestones: list[int],
    gamma: float,
    test_loader: Optional[DataLoader] = None,
    validate: Optional[bool] = False,
    save_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    **kwargs: int,
) -> None:
    """
    Trains model and saves it.

    Args:
        train_loader: self-explanatory
        lr: initial learning rate
        total_epochs: self-explanatory
        milestones: id of epochs where to decrease lr by gamma
        gamma: factor by which to decrease lr
        test_loader: self-explanatory
        validate: whether to validate after every epoch
        save_path: relative path (from directory) where to save
        checkpoint: relative path with model to start with
        kwargs: arguments to initialize the model
    """
    model = pl_class.model_architecture(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1*lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    plmodel = pl_class(model, optimizer, scheduler)
    trainer = pl.Trainer(max_epochs=total_epochs)

    val_args = {"val_dataloaders": test_loader} if validate else {}
    trainer.fit(plmodel, train_loader, **val_args)  # type: ignore
    if test_loader:
        trainer.test(plmodel, test_loader)

    if save_path:
        torch.save(model.state_dict(), save_path)


def plot_images(
    images: torch.Tensor,
    figsize: tuple[float, float] = (12, 4)
):
    """plot hxw array of images"""
    h, w = len(images), len(images[0])
    fig, ax = plt.subplots(h, w, figsize=figsize)
    axs: list[list[Axes]]
    if h == 1:
        axs = [ax]  # type: ignore
    else:
        axs = ax  # type: ignore

    for i in range(h):
        for j in range(w):
            axs[i][j].imshow(images[i][j], cmap='gray')
            axs[i][j].axis('off')
    plt.tight_layout()
    plt.show()


def generate_images(
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
    NN models need not be in device.

    Args:
        width: number of images to have horizontally
        height: number of images to produce vertically
        flow_nn: flow model. Space must match with input of decoder
        autoencoder: autoencoder
        labels: if provided, list of length width*height with labels.
            None refers to no labels for that particular image.
        w: weight of the condition for flow.
        sigma_fn: diffussion coefficient.
        scale: size of each image to plot.
        num_steps: number of steps in integration
    """
    n_imgs = height * width
    z_shape = autoencoder.model.z_shape
    autoencoder.to(model.device)
    autoencoder.eval()

    ones = torch.empty((n_imgs, *z_shape), device=model.device, dtype=torch.float32)
    z0 = torch.normal(torch.zeros_like(ones), torch.ones_like(ones))

    with torch.no_grad():
        z1 = integrate_flow(
            model, z0, labels, w, noise_fn, num_steps, t0=0, t1=1, full_output=False
        )
        x1 = autoencoder.model.decode(z1)
        imgs = torch.clip(x1.detach().cpu(), 0, 1)  # (n_imgs, 1, 28, 28)

    plot_images(
        imgs.view(height, width, *imgs.shape[2:]),
        figsize=(scale*width, scale*height),
    )
