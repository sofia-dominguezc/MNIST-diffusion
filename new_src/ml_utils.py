import os
from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import nn, Tensor


def dummy_noise(t: Tensor) -> Tensor:
    return Tensor(1.0)


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
    merge = not split == "byclass"

    def _name(idx: int) -> str:
        """Maps an index in [0, 46] or [0, 61] to a str"""
        if merge:
            return merged_EMNIST_names[idx]
        else:
            return unmerged_EMNIST_names[idx]

    return [_name(cast(int, idx.item())) for idx in y]


def EMNIST_get_label(
    names: list[str],
    split: Literal['balanced', 'byclass', 'bymerge'] = 'balanced'
) -> Tensor:
    """Convert list of names to 1d tensor of labels"""
    merge = not split == "byclass"

    def _label(name: str) -> int:
        """Maps a str of length 1 to index in [0, 46] or [0, 61]"""
        if merge:
            try:
                return merged_EMNIST_labels[name]
            except KeyError:
                return merged_EMNIST_labels[name.upper()]
        else:
            return unmerged_EMNIST_labels[name]

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


def load_model(
    model_architecture: type[nn.Module],
    model_version: Optional[Literal["dev", "main"]] = None,
    root: str = "parameters",
    **nn_kwargs: int,
) -> nn.Module:
    """
    Initialize, load parameters, and return the specified nn.Module
    kwargs: arguments to initialize the model
    """
    model = model_architecture(**nn_kwargs)
    if model_version is None:
        return model

    name = model_architecture.__name__
    if model_version == "dev":
        path = os.path.join(root, f"{name}_dev.pth")
    else:
        path = os.path.join(root, f"{name}.pth")

    model.load_state_dict(torch.load(path))
    return model


def save_model(
    model: nn.Module,
    model_version: Literal["dev", "main"],
    root: str = "parameters",
):
    if model_version == "dev":
        path = os.path.join(root, f"{model.__class__.__name__}_dev.pth")
    else:
        path = os.path.join(root, f"{model.__class__.__name__}.pth")
    torch.save(model.state_dict(), path)
