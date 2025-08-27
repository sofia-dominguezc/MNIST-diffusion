from typing import Any, Optional, Literal

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import architectures as architectures
from ml_utils import save_model


class GeneralModel(pl.LightningModule):
    """Parent class for AE, VAE, Flow that implements the train/test loop"""
    model_architecture: type[nn.Module]
    loss_kwargs: dict[str, float | int] = {}

    def __init__(
        self,
        model: nn.Module,
        dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler

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
        print(f"\nEpoch {self.current_epoch} - loss: {loss:.4f}")
        save_model(self.model, dataset=self.dataset, model_version='dev')

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


class PlAutoEncoder(GeneralModel):
    model_architecture = architectures.AutoEncoder
    model: architectures.AutoEncoder
    loss_kwargs = {"norm": 1}

    def _loss(
        self, x: Tensor, y: Tensor, norm: float = 1, **kwargs,
    ) -> Tensor:
        """L_p reconstruction loss. x: (batch, *dims)"""
        pred_x = self.model.decode(self.model.encode(x))
        return torch.sum(torch.abs(pred_x - x)**norm) / x.shape[0]


class PlVarAutoEncoder(GeneralModel):
    model_architecture = architectures.VarAutoEncoder
    model: architectures.VarAutoEncoder
    loss_kwargs = {"alpha": 1}

    def _pre_loss(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """compute both losses separately. x: (batch, *dims)"""
        z_mean, z_var = self.model.get_encoding(x)
        ep = torch.normal(torch.zeros_like(z_mean), torch.ones_like(z_mean))
        z = z_mean + torch.sqrt(z_var) * ep
        x_mean, x_var = self.model.get_decoding(z)

        mse_loss = 0.5 * torch.sum((x_mean - x)**2 / x_var)
        kl_loss = 0.5 * torch.sum(z_mean**2 + z_var - 1 - torch.log(z_var))
        return mse_loss / x.shape[0], kl_loss / x.shape[0]  # avg over batches

    def _loss(
        self, x: Tensor, y: Tensor, alpha: float = 1, **kwargs,
    ) -> Tensor:
        mse_loss, kl_loss = self._pre_loss(x)
        return mse_loss + alpha * kl_loss

    def _acc(
        self, x: Tensor, y: Tensor, **kwargs,
    ) -> dict[str, Tensor]:
        mse_loss, kl_loss = self._pre_loss(x)
        return {"mse_loss": mse_loss, "kl_loss": kl_loss}


class PlDiffusion(GeneralModel):
    model_architecture = architectures.Diffusion
    model: architectures.Diffusion
    loss_kwargs = {"prob": 0.1}

    loss_fn = nn.MSELoss(reduction='mean')

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss_kwargs["n_classes"] = self.model.n_classes

    def _process_labels(
        self, y: Tensor, n_classes: int, prob: float
    ) -> Tensor:
        """Converts y to one_hot and erases labels with probability prob"""
        new_y = nn.functional.one_hot(
            y.to(torch.long), num_classes=n_classes
        ).to(torch.float32)
        unif = torch.rand_like(
            y.to(torch.float32)
        ).view(-1, 1).repeat(1, n_classes)
        return torch.where(unif < prob, torch.zeros_like(new_y), new_y)

    def _sample_noise(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        """Returns x0 ~ N(0, I) and t ~ Unif[0, 1]"""
        x0 = torch.normal(torch.zeros_like(x1), torch.ones_like(x1))
        t_shape = [x1.shape[0]] + [1] * (len(x1.shape) - 1)
        t = torch.rand(t_shape, device=x1.device)
        return x0, t

    def _loss(
        self,
        x: Tensor,
        y: Tensor,
        n_classes: int = 47,
        prob: float = 0.1,
        **kwargs,
    ) -> Tensor:
        """loss function for flow/diffusion. x: (batch, *dims)"""
        y = self._process_labels(y, n_classes=n_classes, prob=prob)
        x0, t = self._sample_noise(x)
        xt = (1 - t) * x0 + t * x
        pred_vf = self.model(xt, t, y)
        return self.loss_fn(pred_vf, x - x0)


def train(
    model: nn.Module,
    pl_class: type[GeneralModel],
    dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
    train_loader: DataLoader,
    lr: float,
    total_epochs: int,
    milestones: list[int],
    gamma: float,
    test_loader: Optional[DataLoader] = None,
):
    """
    Trains model and saves it.

    Args:
        train_loader: self-explanatory
        lr: initial learning rate
        total_epochs: self-explanatory
        milestones: id of epochs where to decrease lr by gamma
        gamma: factor by which to decrease lr
        test_loader: self-explanatory
        save_path: relative path (from root) where to save
        root: path with all model parameters
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1*lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    plmodel = pl_class(model, dataset, optimizer=optimizer, scheduler=scheduler)
    trainer = pl.Trainer(max_epochs=total_epochs, logger=False, enable_checkpointing=False)

    val_args = {"val_dataloaders": test_loader} if test_loader else {}
    trainer.fit(plmodel, train_loader, **val_args)  # type: ignore
    if test_loader:
        trainer.test(plmodel, test_loader)


def test(
    model: nn.Module,
    pl_class: type[GeneralModel],
    dataset: Literal["MNIST", "EMNIST", "FashionMNIST"],
    test_loader: DataLoader,
):
    plmodel = pl_class(model, dataset)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(plmodel, test_loader)
