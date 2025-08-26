import torch
from torch import nn, Tensor
import pytorch_lightning as pl
import new_src.architectures as architectures
from typing import Any


class GeneralModel(pl.LightningModule):
    """Parent class for AE, VAE, Flow that implements the train/test loop"""
    model_architecture: type[nn.Module]
    model: nn.Module

    def __init__(self, optimizer, scheduler, **kwargs):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = self.model_architecture(**kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _loss(self, x: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def _acc(self, x: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """By default use training metric for testing"""
        return {"loss": self._loss(x, **kwargs)}

    def training_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int
    ) -> Tensor:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = self._loss(x)
        self.log("loss_step", loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics["loss_epoch"]
        print(f"Epoch {self.current_epoch} - loss: {loss:.4f}")

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int
    ):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        accs = self._acc(x)
        for name, acc in accs.items():
            self.log(name, acc, on_epoch=True, prog_bar=False)


class AutoEncoder(GeneralModel):
    model_architecture = architectures.AutoEncoder
    model: architectures.AutoEncoder

    def _loss(self, x: Tensor, norm: float = 1) -> Tensor:
        """L_p reconstruction loss. x: (batch, *dims)"""
        pred_x = self.model.decode(self.model.encode(x))
        return torch.sum(torch.abs(pred_x - x)**norm) / x.shape[0]


class VarAutoEncoder(GeneralModel):
    model_architecture = architectures.VarAutoEncoder
    model: architectures.VarAutoEncoder

    def _pre_loss(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """compute both losses separately. x: (batch, *dims)"""
        z_mean, z_var = self.model.get_encoding(x)
        ep = torch.normal(torch.zeros_like(z_mean), torch.ones_like(z_mean))
        z = z_mean + torch.sqrt(z_var) * ep
        x_mean, x_var = self.model.get_decoding(z)

        mse_loss = 0.5 * torch.sum((x_mean - x)**2 / x_var)
        kl_loss = 0.5 * torch.sum(z_mean**2 + z_var - 1 - torch.log(z_var))
        return mse_loss / x.shape[0], kl_loss / x.shape[0]  # avg over batches

    def _loss(self, x: Tensor, alpha: float = 1) -> Tensor:
        mse_loss, kl_loss = self._pre_loss(x)
        return mse_loss + alpha * kl_loss

    def _acc(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        mse_loss, kl_loss = self._pre_loss(x)
        return {"mse_loss": mse_loss, "kl_loss": kl_loss}


class Diffusion(GeneralModel):
    model_architecture = architectures.Diffusion
    model: architectures.Diffusion

    loss_fn = nn.MSELoss(reduction='mean')
    prob = 0.1
    n_classes = 47

    def _process_labels(self, y: Tensor, n_classes: int, prob: float) -> Tensor:
        """Converts y to one_hot and erases labels with probability prob"""
        new_y = nn.functional.one_hot(
            y.to(torch.long), num_classes=n_classes
        ).to(torch.float32)
        unif = torch.rand_like(y.to(torch.float32)).view(-1, 1).repeat(1, n_classes)
        return torch.where(unif < prob, torch.zeros_like(new_y), new_y)

    def _sample_noise(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        """Returns x0 ~ N(0, I) and t ~ Unif[0, 1]"""
        x0 = torch.normal(torch.zeros_like(x1), torch.ones_like(x1))
        t_shape = tuple(dsize if i == 0 else 1 for i, dsize in enumerate(x1.shape))
        t = torch.rand(t_shape, device=x1.device)
        return x0, t

    def _loss(self, x1: Tensor, y: Tensor) -> Tensor:
        """loss function for flow/diffusion. x: (batch, *dims)"""
        y = self._process_labels(y, n_classes=self.n_classes, prob=self.prob)
        x0, t = self._sample_noise(x1)
        xt = (1 - t) * x0 + t * x1
        pred_vf = self.model(xt, t, y)
        return self.loss_fn(pred_vf, x1 - x0)
