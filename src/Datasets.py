import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional
from typing import Union, Callable

class TransposeTransform(nn.Module):
    def forward(self, img, label=None):
        return img.transpose(-1, -2)

def load_EMNIST(
    root: str,
    train: Optional[bool] = True,
    split: Optional[str] = 'balanced',
    batch_size: Optional[int] = 128,
    num_workers: Optional[int] = 0,
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
    # define arguments
    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    } if num_workers > 0 else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeTransform(),
    ])
    # load dataset and dataloader
    dataset = datasets.EMNIST(
        root=root, train=train, download=False, transform=transform, split=split,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, **workers_args,
    )
    return dataloader

def load_TensorDataset(
    root: str,
    batch_size: Optional[int] = 128,
    shuffle: Optional[bool] = False,
    num_workers: Optional[int] = 0,
) -> DataLoader:
    """
    Loads file stored in .py files.
    features data is in root/dataset_x.pt
    labels data is in root/dataset_y.pt

    Args:
        root: path (from directory) to folder with dataset
        batch_size: batch_size for dataloader
        shuffle: whether to shuffle the dataloader
        num_workers: parallelize computation. If num_workers > 0, then
            pin_memory=True and persistent_workers=True

    Returns:
        dataloader with the saved dataset
    """
    torch.cuda.empty_cache()
    # define arguments
    workers_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    } if num_workers > 0 else {}
    # load dataset
    path_x = os.path.join(root, "dataset_x.pt")
    path_y = os.path.join(root, "dataset_y.pt")
    tensor_x = torch.load(path_x)
    tensor_y = torch.load(path_y)
    dataset = TensorDataset(tensor_x, tensor_y)
    # return dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, **workers_args,
    )
    return dataloader

def process_dataset(
    data: Union[Dataset, DataLoader],
    save_path: str,
    fn_x: Callable,
    fn_y: Callable,
    shape_x: tuple[int],
    shape_y: tuple[int],
    batch_size: Optional[int] = 128,
    device: Optional[str] = 'cuda',
) -> None:
    """
    Process a dataset according to given functions.
    Store tensors in save_path/dataset_x (or y)

    Args:
        data: original dataset/dataloader to process
        save_path: relative path to save tensors
        fn_x: function to apply to batched inputs (on device)
        fn_y: function to apply to batched labels
        shape_x: shape of a processed x tensor
        shape_y: shape of a processed y tensor
        batch_size: batch_size to use if data is a Dataset
    """
    # make dataloader
    if not isinstance(data, DataLoader):
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        dataloader = data
    # evaluation loop
    newdata_x = torch.zeros(len(dataloader.dataset), *shape_x)
    newdata_y = torch.zeros(len(dataloader.dataset), *shape_y)
    with torch.no_grad():
        idx = 0
        for x, y in dataloader:
            B = x.shape[0]
            x, y = x.to(device), y.to(device)
            new_x = fn_x(x, y).detach().cpu()
            new_y = fn_y(x, y).detach().cpu()
            newdata_x[idx: idx + B] = new_x
            newdata_y[idx: idx + B] = new_y
            idx += B
    # save tensors
    if save_path:
        path_x = os.path.join(save_path, "dataset_x.pt")
        path_y = os.path.join(save_path, "dataset_y.pt")
        torch.save(newdata_x, path_x)
        torch.save(newdata_y, path_y)
