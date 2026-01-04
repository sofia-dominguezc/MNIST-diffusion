import torch
from torch import Tensor

from utils import GeneralModel
from vae.architecture import AutoEncoder, VarAutoEncoder


class PlAutoEncoder(GeneralModel):
    model_architecture = AutoEncoder
    model: AutoEncoder
    loss_kwargs = {"norm": 1}

    def _loss(
        self, x: Tensor, y: Tensor, norm: float = 1, **kwargs,
    ) -> Tensor:
        """L_p reconstruction loss. x: (batch, *dims)"""
        pred_x = self.model.decode(self.model.encode(x))
        return torch.sum(torch.abs(pred_x - x)**norm) / x.shape[0]


class PlVarAutoEncoder(GeneralModel):
    model_architecture = VarAutoEncoder
    model: VarAutoEncoder
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
