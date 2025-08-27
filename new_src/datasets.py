import os
from tqdm import tqdm
from typing import Union, Optional

import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset

from architectures import AutoEncoder, VarAutoEncoder


class TransposeTransform(nn.Module):
    def forward(self, img, label=None):
        return img.transpose(-1, -2)


def load_MNIST() -> DataLoader: ...


def load_FashionMNIST() -> DataLoader: ...


def load_EMNIST(
    root: str = "data",
    train: bool = True,
    split: str = 'balanced',
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """
    Loads and returns dataloader of EMNIST.
    Shuffles the dataloader iff mode is train.

    Args:
        root: path (from directory) to file where dataset lies
        train: whether to use train or test dataset
        split: 'balanced', 'byclass', or 'bymerge'
        batch_size: batch_size for dataloader
        num_workers: if num_workers > 0, then
            pin_memory=True and persistent_workers=True

    Returns:
        dataloader with the saved dataset
    """
    torch.cuda.empty_cache()

    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    } if num_workers > 0 else {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeTransform(),
    ])

    dataset = datasets.EMNIST(
        root=root, train=train, download=False, transform=transform, split=split,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, **workers_args,
    )


def encode_dataset(
    data: Union[Dataset, DataLoader],
    autoencoder: Union[AutoEncoder, VarAutoEncoder],
    save_path: Optional[str] = None,
    root: str = "data",
    batch_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """
    Creates dataset with features (autoencoder.encode(x), y)
    and saves it under root/save_path/ as z.pt, y.pt
    """
    if not isinstance(data, DataLoader):
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        dataloader = data
    device = next(autoencoder.parameters()).device

    n_batch = len(dataloader.dataset)  # type: ignore
    data_z = torch.empty(n_batch, *autoencoder.z_shape)
    data_y = torch.empty(n_batch)
    with torch.no_grad():
        idx = 0
        for x, y in tqdm(dataloader):
            batches = x.shape[0]
            z = autoencoder.encode(x.to(device)).detach().cpu()
            data_z[idx: idx + batches] = z
            data_y[idx: idx + batches] = y
            idx += batches

    if save_path:
        folder = os.path.join(root, save_path)
        os.makedirs(folder, exist_ok=True)
        path_z = os.path.join(folder, "z.pt")
        path_y = os.path.join(folder, "y.pt")
        torch.save(data_z, path_z)
        torch.save(data_y, path_y)

    return data_z, data_y


def load_TensorDataset(
    data_path: str,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    root: str = "data",
    names: list[str] = ["z.pt", "y.pt"]
) -> DataLoader:
    """
    Loads specified files in root/save_path/

    Args:
        save_path: path from data folder to files folder
        batch_size: batch_size for dataloader
        shuffle: whether to shuffle the dataloader
        num_workers: parallelize computation. If num_workers > 0, then
            pin_memory=True and persistent_workers=True
        root: path from directory to data folder
        names: list of file names to retrieve

    Returns:
        dataloader with the saved dataset
    """
    torch.cuda.empty_cache()
    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    } if num_workers > 0 else {}

    data = []
    for name in names:
        path = os.path.join(root, data_path, name)
        tensor = torch.load(path)
        data.append(tensor)
    dataset = TensorDataset(*data)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **workers_args,
    )
    return dataloader
