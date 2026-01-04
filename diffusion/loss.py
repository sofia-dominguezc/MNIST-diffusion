import torch
import torch.nn as nn
import lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Any, Optional, Literal

from diffusion.architecture import Diffusion
from vae.architecture import VarAutoEncoder, AutoEncoder
from utils import GeneralModel, sample_noise


class PlFlow(GeneralModel):
    model_architecture = Diffusion
    model: Diffusion
    loss_kwargs = {"prob": 0.1}

    loss_fn = nn.MSELoss(reduction='mean')

    def __init__(
        self,
        *args: Any,
        noise_type: Literal["gaussian", "uniform", "bernoulli"] = "gaussian",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.noise_type = noise_type

    def _process_labels(self, y: Tensor, prob: float) -> Tensor:
        unif = torch.rand_like(y.float())
        return torch.where(unif < prob, - torch.ones_like(y), y)

    def _sample_noise(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        """Returns x0 ~ p_noise and t ~ Unif[0, 1]"""
        x0 = sample_noise(self.noise_type, x1.shape, x1.device, x1.dtype)

        t_shape = (x1.shape[0], *(1, ) * (x1.ndim - 1))
        s = -0.5 + 0.8 * torch.randn(t_shape, device=x1.device)
        t = torch.sigmoid(s)
        return x0, t

    def _loss(
        self,
        x: Tensor,
        y: Tensor,
        prob: float = 0.1,
        **kwargs,
    ) -> Tensor:
        """loss function for flow/diffusion. x: (batch, *dims)"""
        y = self._process_labels(y, prob=prob)
        x0, t = self._sample_noise(x)

        xt = t * x + (1 - t) * x0
        true_vf = x - x0

        pred_x1 = self.model(xt, t, y)
        pred_vf = (pred_x1 - xt) / (1 - t)

        return self.loss_fn(pred_vf, true_vf)


class PlClassifier(pl.LightningModule):
    """
    Model that implements classification using Bayes Rule:
            p(y | x) ~ p(x | y)p(y)
    """
    def __init__(
        self,
        model: Diffusion,
        autoencoder: AutoEncoder | VarAutoEncoder,
        num_steps: int = 50,
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder
        self.num_steps = num_steps

    def _classify(
        self,
        model: Diffusion,
        autoencoder: AutoEncoder | VarAutoEncoder,
        x: Tensor,
        num_steps: int = 100,
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

    def _acc(self, x: Tensor, y: Tensor) -> float:
        """Find top1 and top2 prediction accuracy"""
        logits = self._classify(  # (batch, n_class)
            self.model,
            self.autoencoder,
            x,
            num_steps=self.num_steps,
        )
        predictions = torch.argmax(logits, dim=-1)  # (batch, )
        acc = (predictions == y).sum() / y.shape[0]
        return 100 * acc.item()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        acc = self._acc(x, y)
        self.log("accuracy", acc, on_epoch=True, prog_bar=True)


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
    alpha: float = 1,
    **pl_kwargs: Any,
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
        alpha: weight of the KL loss compared to MSE for VAEs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1*lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)

    plmodel = pl_class(model, dataset, optimizer=optimizer, scheduler=scheduler, **pl_kwargs)
    if "alpha" in plmodel.loss_kwargs:
        plmodel.loss_kwargs["alpha"] = alpha
    trainer = pl.Trainer(max_epochs=total_epochs, logger=False, enable_checkpointing=False)

    val_args = {"val_dataloaders": test_loader} if test_loader else {}
    trainer.fit(plmodel, train_loader, **val_args)  # type: ignore
    if test_loader:
        trainer.test(plmodel, test_loader)


def test_classification(
    flow: Diffusion,
    autoencoder: AutoEncoder | VarAutoEncoder,
    test_loader: DataLoader,
    num_steps: int = 50,
):
    raise NotImplementedError
    classifier = PlClassifier(flow, autoencoder, num_steps=num_steps)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(classifier, test_loader)
