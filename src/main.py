import os
from dotenv import load_dotenv
import random
import torch
from torch import nn
from torch import optim
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Optional, Callable, Tuple, Union

device = 'cuda'
batch_size = 512
split = 'balanced'  # for EMNIST
num_classes = 47

def train(
    model: torch.nn.Module,
    PLClass: pl.LightningModule,
    train_loader: DataLoader,
    lr: float,
    total_epochs: int,
    milestones: List[int],
    gamma: float,
    test_loader: Optional[DataLoader] = None,
    validate: Optional[bool] = False,
    save_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> None:
    """
    Trains model and saves it.

    Args:
        model: nn.Module to train
        PLClass: LightningModule class to use
        train_loader: self-explanatory
        lr: initial learning rate
        total_epochs: self-explanatory
        milestones: id of epochs where to decrease lr by gamma
        gamma: factor by which to decrease lr
        test_loader: self-explanatory
        validate: whether to validate after every epoch
        save_path: relative path (from directory) where to save
        checkpoint: relative path with model to start with
    """
    # Load checkpoint
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
    # Preparations
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1*lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    plmodel = PLClass(model, optimizer, scheduler)
    trainer = pl.Trainer(max_epochs=total_epochs)
    # Train and test
    val_args = {"val_dataloaders": test_loader} if validate else {}
    trainer.fit(plmodel, train_loader, **val_args)
    if test_loader:
        trainer.test(plmodel, test_loader)
    # Save model
    if save_path:
        torch.save(model.state_dict(), save_path)

def soft_labels(
        data: Union[Dataset, DataLoader],
        classifier: nn.Module,
        save_path: str,
    ):
    """
    Creates a datset of soft labels from the data using
    the provided classifier.
    Store tensors in save_path/dataset_x (or y).

    Args:
        data: dataset or dataloader with original data
        classifier: forward method must output logits
        save_path: path to save processed dataset
    """
    process_dataset(
        data,
        save_path,
        fn_x=lambda x, y: x,
        fn_y=lambda x, y: classifier(x).softmax(dim=-1),
        shape_x=(1, 28, 28),
        shape_y=(classifier.num_classes, ),
        device=device,
    )

def encode_datset(
        data: Union[Dataset, DataLoader],
        autoencoder: nn.Module,
        save_path: str,
    ):
    """
    Creates a datset of encoded features from the data
    using the provided autoencoder.
    Store tensors in save_path/dataset_x (or y).

    Args:
        data: dataset or dataloader with original data
        autoencoder: must implement self.encode(x)
        save_path: path to save processed dataset
    """
    process_dataset(
        data,
        save_path,
        fn_x=lambda x, y: autoencoder.encode(x),
        fn_y=lambda x, y: y,
        shape_x=autoencoder.z_shape,
        shape_y=(),
        device=device,
    )

def plot_ae_reconstruction(
        autoencoder: nn.Module,
        dataloader: DataLoader,
        width=10,
        height=2,
        scale=1.4,
    ):
    """
    Plots images and their reconstruction by the autoencoder.
    Assumes num_img < batch_size

    Args:
        autoencoder: model to use
        data: where to get images from
        width: how many images horizontally (along with reconstructions)
        height: how many images to plot vertically
    """
    num_img = width * height
    # choose batch
    num_batches = len(dataloader)
    batch_idx = random.choice(range(num_batches))
    for i, (x, y) in enumerate(dataloader):
        if i >= batch_idx:
            break
    # choose images
    list_imgs = random.choices(x, k=num_img)
    imgs = torch.stack(list_imgs).to(device)
    # pass through the model
    autoencoder.to(device)
    imgs_hat = autoencoder(imgs)
    # reshape and move to numpy
    imgs = imgs[:, 0].cpu().detach()  # (num_img, 28, 28)
    imgs = imgs.view(height, width, *imgs.shape[-2:])
    imgs_hat = torch.clip(imgs_hat[:, 0].cpu().detach(), 0, 1)
    imgs_hat = imgs_hat.view(height, width, *imgs_hat.shape[-2:])
    # combine along width axis
    combined_imgs = torch.empty((height, 2*width, *imgs.shape[2:]))
    combined_imgs[:, ::2] = imgs
    combined_imgs[:, 1::2] = imgs_hat
    # plot
    plot_images(
        combined_imgs,
        figsize=(2*scale*width, scale*height),
    )

def generate_flow_images(
        width: int,
        height: int,
        flow_nn: nn.Module,
        autoencoder: nn.Module,
        labels: Optional[List[Union[int, None]]] = None,
        w: Optional[float] = 1,
        sigma_fn: Optional[Callable] = lambda t: torch.sqrt(1 - t),
        scale: Optional[float] = 1,
        num_steps: Optional[int] = 50,
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
    n_class = flow_nn.num_classes
    n_imgs = height * width
    z_shape = autoencoder.z_shape
    autoencoder.to(device)
    autoencoder.eval()
    # sample random noise
    ones = torch.empty((n_imgs, *z_shape), device=device, dtype=torch.float32)
    z0 = torch.normal(torch.zeros_like(ones), torch.ones_like(ones))
    # process labels
    if labels is not None:
        labels = [k if k is not None else n_class for k in labels]
        y = torch.tensor(labels, device=device)
        y = one_hot(y, num_classes=n_class+1).to(torch.float32)
        y = y[:, :-1]  # delete None label
    else:
        y = torch.zeros((n_imgs, n_class), dtype=torch.float32, device=device)
    # solve SDE and recreate images
    with torch.no_grad():
        z1 = forward_flow(flow_nn, z0, y, w, sigma_fn, num_steps)
        x1 = autoencoder.decode(z1)
        imgs = torch.clip(x1.detach().cpu(), 0, 1)  # (n_imgs, 1, 28, 28)
    # plot genereated images
    plot_images(
        imgs.view(height, width, *imgs.shape[2:]),
        figsize=(scale*width, scale*height),
    )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    from Datasets import load_EMNIST, load_TensorDataset, process_dataset
    from EMNISTModels import AutoEncoder, FlowMatching, Classifier
    from Train import FlowModel, AutoEncoderModel, FlowClassifierModel, ClassifierModel
    from Utils import forward_flow, backward_flow, plot_images
    load_dotenv()
    print("Finished importing packages")

    # load EMNIST
    root_path = os.getenv("NIST_ROOT_PATH")
    train_loader = load_EMNIST(
        root_path, True, split, batch_size, num_workers=0,
    )
    test_loader = load_EMNIST(
        root_path, False, split, batch_size, num_workers=8,
    )

    # define model architecture
    classifier = Classifier(
        d1=16, d2=24, k=5, n=1, n_heads=2, head_dim=8,
        n_class=num_classes,
    ).to(device)
    classifier_checkpoint = os.getenv("EMNIST_CLASSIFIER")
    # classifier.load_state_dict(torch.load(classifier_checkpoint))

    # autoencoder = AutoEncoder(
    #     d1=24, d2=32, k=4, n=1, n_heads=3, head_dim=8,
    #     p_norm=1,
    #     # sigma=0.2,
    # ).to(device)
    # autoencoder_checkpoint = os.getenv("EMNIST_AUTOENCODER_L1")
    # autoencoder.load_state_dict(torch.load(autoencoder_checkpoint))

    # flow_nn = FlowMatching(
    #     d=48, k=5, n=2, n_heads=5, head_dim=8, n_class=num_classes,
    #     p=0.2, z_dim=7,
    # ).to(device)
    # flow_checkpoint = os.getenv("EMNIST_FLOWMATCHING")
    # flow_nn.load_state_dict(torch.load(flow_checkpoint))

    # # print(f"Creating encoded datsets...")
    # encoded_train_path = os.path.join(root_path, "encoded_EMNIST_train")
    # encoded_test_path = os.path.join(root_path, "encoded_EMNIST_test")
    # encode_datset(train_loader, autoencoder, encoded_train_path)
    # encode_datset(test_loader, autoencoder, encoded_test_path)

    # # load encoded dataset
    # train_loader = load_TensorDataset(
    #     encoded_train_path,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=16,
    # )
    # test_loader = load_TensorDataset(
    #     encoded_test_path,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,
    # )

    # # train model
    # train(
    #     flow_nn,
    #     FlowModel,
    #     train_loader,
    #     lr=5e-3,
    #     total_epochs=30,
    #     milestones=list(range(15, 101, 6)),
    #     gamma=0.4,
    #     test_loader=test_loader,
    #     validate=False,
    #     # checkpoint=flow_checkpoint,
    #     save_path=flow_checkpoint,
    # )

    # trainer = pl.Trainer()
    # trainer.test(ClassifierModel(classifier, None, None), test_loader)

    # torch.cuda.empty_cache()

    # plot_var_vae(vae, test_loader, device=device)  # sanity check

    height, width, scale, w = 12, 20, 0.8, 3
    # plot_ae_reconstruction(
    #     autoencoder,
    #     test_loader,
    #     width=width // 2,
    #     height=height,
    #     scale=scale,
    # )

    generate_flow_images(
        width=width,
        height=height,
        flow_nn=flow_nn,
        autoencoder=autoencoder,
        labels=random.choices(range(num_classes), k=height*width),
        w=w,
        sigma_fn=lambda t: 1 - t,
        num_steps=150,
        scale=scale,
    )

    # pl_flow_classify = FlowClassifierModel(
    #     flow_nn, classifier, autoencoder
    # )
    # trainer = pl.Trainer()
    # trainer.test(pl_flow_classify, test_loader)
