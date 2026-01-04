import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Literal, Sequence, Any, cast

import torch
import torch.nn as nn
import lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader


def dummy_noise(t: Tensor) -> Tensor:
    return 1 - t


merged_EMNIST_names = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
)
merged_EMNIST_labels = {
    name: idx for idx, name in enumerate(merged_EMNIST_names)
}

unmerged_EMNIST_names = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
unmerged_EMNIST_labels = {
    name: idx for idx, name in enumerate(unmerged_EMNIST_names)
}


def EMNIST_get_name(
    y: Tensor,
    split: Literal['balanced', 'byclass', 'bymerge'] = 'balanced'
) -> list[str]:
    """Convert 1d tensor of labels to list of names"""
    assert len(y.shape) == 1, "Tensor is not 1D"
    assert not torch.is_floating_point(y), "Tensor must contain int"

    def _name(idx: int) -> str:
        """Maps an index in [0, 46] or [0, 61] to a str"""
        if split == "byclass":
            return unmerged_EMNIST_names[idx]
        else:
            return merged_EMNIST_names[idx]

    return [_name(cast(int, idx.item())) for idx in y]


def EMNIST_get_label(
    names: list[str],
    split: Literal['balanced', 'byclass', 'bymerge'] = 'balanced'
) -> Tensor:
    """Convert list of names to 1d tensor of labels"""
    def _label(name: str) -> int:
        """Maps a str of length 1 to index in [0, 46] or [0, 61]"""
        if split == "byclass":
            return unmerged_EMNIST_labels[name]
        else:
            try:
                return merged_EMNIST_labels[name]
            except KeyError:
                return merged_EMNIST_labels[name.upper()]

    return Tensor([_label(name) for name in names])


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


def get_num_classes(
    dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
    split: Literal["balanced", "byclass", "bymerge"]
) -> int:
    if dataset != "EMNIST":
        return 10
    if split == "byclass":
        return len(unmerged_EMNIST_names)
    else:
        return len(merged_EMNIST_names)


def load_model(
    model_architecture: type[nn.Module],
    dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
    model_version: Literal["dev", "main"] | None = None,
    root: str = "parameters",
    **nn_kwargs: int,
) -> nn.Module:
    """
    Initialize, load parameters, and return the specified nn.Module
    kwargs: arguments to initialize the model, used if model_version=None
    """
    print(f"Trying to load {model_architecture.__name__}-{model_version}...", end=' ')
    if model_version is None:
        return model_architecture(**nn_kwargs)

    name = model_architecture.__name__
    if model_version == "dev":
        path = os.path.join(root, "dev", dataset, name)
    else:
        path = os.path.join(root, dataset, name)

    try:
        with open(f"{path}.pickle", "rb") as f:
            init_args = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameters for {path} were not found")

    try:
        model = model_architecture(**init_args)
    except TypeError:
        raise TypeError(
            f"Saved metadata for {path} doesn't match current class implementation"
        )
    model.load_state_dict(torch.load(f"{path}.pth"))
    print("Success!")
    return model


def save_model(
    model: nn.Module,
    dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
    model_version: Literal["dev", "main"],
    root: str = "parameters",
):
    name = model.__class__.__name__
    if model_version == "dev":
        folder = os.path.join(root, "dev", dataset)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
    else:
        folder = os.path.join(root, dataset)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)

    torch.save(model.state_dict(), f"{path}.pth")
    with open(f"{path}.pickle", "wb") as f:
        pickle.dump(model.init_args, f)


class GeneralModel(pl.LightningModule):
    """Parent class for AE, VAE, Flow that implements the train/test loop"""
    model_architecture: type[nn.Module]
    loss_kwargs: dict[str, float | int] = {}

    def __init__(
        self,
        model: nn.Module,
        dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.MultiStepLR | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lowest_train_loss = float('inf')

    def configure_optimizers(self):  # type: ignore
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _loss(self, x: Tensor, y: Tensor, **kwargs) -> Tensor: ...

    def _acc(self, x: Tensor, y: Tensor, **kwargs) -> dict[str, Tensor]:
        """By default use training metric for testing"""
        return {"loss": self._loss(x, y, **kwargs)}

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int,
    ) -> Tensor:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = self._loss(x, y, **self.loss_kwargs)
        self.log("loss_step", loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics["loss_epoch"]
        print(f"\nEpoch {self.current_epoch} - loss_train: {loss:.4f}")
        if loss < self.lowest_train_loss:
            self.lowest_train_loss = loss
            save_model(
                self.model,
                dataset=self.dataset,  # type: ignore
                model_version='dev',
            )

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        accs = self._acc(x, y, **self.loss_kwargs)
        for name, acc in accs.items():
            self.log(f"{name}_val", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        accs = self._acc(x, y, **self.loss_kwargs)
        for name, acc in accs.items():
            self.log(name, acc, on_epoch=True, prog_bar=False)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)


def sample_noise(
    noise_type: str,
    shape: Sequence[int],
    device: str | torch.device,
    dtype: torch.dtype,
):
    if noise_type == "gaussian":
        return  torch.normal(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.ones(shape, device=device, dtype=dtype),
        )
    elif noise_type == "uniform":
        return torch.rand(shape, device=device, dtype=dtype)
    elif noise_type == "bernoulli":
        return torch.randint(0, 2, shape, device=device).to(dtype)
    raise ValueError(f"Invalid {noise_type=}")


def parse_unknown_args(unknown: list[str]) -> dict[str, int]:
    """
    Pass all unknown arguments into the NN
    This feature is only used in train mode
    """
    nn_kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-").replace("-", "_")
        val = unknown[i + 1]
        nn_kwargs[key] = int(val)
    return nn_kwargs


# deprecated -----------------------------


def autoencoder_reconstruction(
    autoencoder,
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
    model,
    autoencoder,
    x: Tensor,
    num_steps: int = 50,
) -> Tensor:
    """
    Classify x using the flow model and autoencoder
    Uses Bayes rule: p(y | x) ~ sum_y p(x | y) p(y)
    NOTE: doesn't work for splits other than "balanced" in EMNIST
          because it assumes p(y) is constant
    """
    n_classes = model.n_classes
    batches = x.shape[0]
    z = autoencoder.encode(x)
    z1 = z.unsqueeze(0).repeat(n_classes, *([1] * z.ndim))  # (n_class, batch, *z_shape)

    labels = process_labels(list(range(n_classes)), n_classes, z.device)  # (n_class, n_class)
    y = labels.unsqueeze(1).repeat(1, batches, 1)  # (n_class, batch, n_class)

    solver = SDESolver(model, y.flatten(0, 1), weight=1, diffusion=0)
    z0_flat = solver.solve(z1.flatten(0, 1), t0=1, t1=0, num_steps=num_steps)
    z0 = z0_flat.unflatten(0, (n_classes, batches)).flatten(2)  # (n_class, batches, prod(z_shape))

    logits = - 0.5 * torch.sum(z0**2, dim=-1)  # (n_class, batches)
    return logits

