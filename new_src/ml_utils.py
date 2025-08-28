import os
import pickle
from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import nn, Tensor


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
    dataset: str,
    split: Literal["balanced", "byclass", "bymerge"] = "balanced",
    model_version: Optional[Literal["dev", "main"]] = None,
    root: str = "parameters",
    **nn_kwargs: int,
) -> nn.Module:
    """
    Initialize, load parameters, and return the specified nn.Module
    kwargs: arguments to initialize the model, used if model_version=None
    """
    if model_version is None:
        nn_kwargs["n_classes"] = 47 if dataset == "EMNIST" else 10
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
        raise FileNotFoundError(
            f"Parameters for {path} were not found. Consider using "
            "the flags --model-version dev or --autoencoder-version dev"
        )

    model = model_architecture(**init_args)
    model.load_state_dict(torch.load(f"{path}.pth"))
    return model


def save_model(
    model: nn.Module,
    dataset: str,
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
