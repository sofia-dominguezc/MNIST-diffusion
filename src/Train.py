import torch
from torch import nn
import pytorch_lightning as pl
import math
from Utils import flow_classify

# Global variables

MSELoss = nn.MSELoss(reduction='mean')
CELoss = nn.CrossEntropyLoss(reduction='mean')

# Train-loop functions

def forward_pass_classification(model, x, y, topk=1):
    y_hat = model(x)  # logits
    loss = CELoss(y_hat, y)  # y_hat: logits. y: labels or probs
    _, indices = torch.topk(y_hat, topk, dim=1)
    if len(y.shape) > 1:  # y: probs
        y = torch.argmax(y, dim=1)
    is_correct = torch.any(indices == y.view(-1, 1), dim=1)
    acc = 100 * torch.mean(is_correct.to(torch.float32))
    return loss, acc

def forward_pass_autoencoding(model, x, y, lp=2):
    """lp: index of the lp norm. 2 is MSE. 1 is L1 norm"""
    x_hat = model.decode(model.encode(x))
    loss = torch.sum(torch.abs(x_hat - x)**lp) / x.shape[0]
    return loss

def forward_pass_var_aut(model, x, y):
    """
    Computes MSE and KL losses. Note that sigma is fixed.
    x: (B, C, H, W)
    z: (B, ...)
    returns (mse_loss, kl_loss)
    """
    B = x.shape[0]
    mu, var = model.get_encoding(x)
    ep = torch.normal(torch.zeros_like(mu), torch.ones_like(mu)).view(mu.shape)
    z = mu + torch.sqrt(var) * ep
    x_hat, sigma = model.get_decoding(z)

    mse_loss = torch.sum((x_hat - x)**2 / sigma) / B
    kl_loss = torch.sum(var - torch.log(var) + mu**2) / B
    return mse_loss, kl_loss  # avg over batches

def sample_x0_t(x1):
    """Randomly samples x0 and t"""
    x0 = torch.normal(torch.zeros_like(x1), torch.ones_like(x1))
    t_shape = tuple(b if i == 0 else 1 for i, b in enumerate(x1.shape))
    t = torch.rand(t_shape, device=x0.device)
    return x0, t

def erase_label(y, p, num_classes):
    """Converts y into one_hot and erases labels
    independently with probability p"""
    unif = torch.rand_like(y.to(torch.float32)).view(-1, 1).repeat(1, num_classes)
    y = nn.functional.one_hot(
        y.to(torch.long), num_classes=num_classes
    ).to(torch.float32)
    return torch.where(unif < p, torch.zeros_like(y), y)

def forward_pass_flow(model, x, y):
    y = erase_label(y, p=model.p, num_classes=model.num_classes)
    x0, t = sample_x0_t(x)
    xt = (1 - t) * x0 + t * x
    vf = x - x0
    vf_hat = model(xt, t, y)
    loss = MSELoss(vf_hat, vf)
    return loss

# Lightning modules

class GeneralModel(pl.LightningModule):
    """Parent class for Classifier, AE, VAE, Flow"""
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

class ClassifierModel(GeneralModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss, acc = forward_pass_classification(self.model, x, y)

        self.log("loss_step", loss, on_epoch=False, prog_bar=True)
        self.log("loss_epoch", loss, on_epoch=True, prog_bar=False)
        self.log("acc_step", acc, on_epoch=False, prog_bar=True)
        self.log("acc_epoch", acc, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        acc = self.trainer.callback_metrics.get("acc_epoch")
        loss = self.trainer.callback_metrics.get('loss_epoch')
        if acc is not None and loss is not None:
            print((
                f"Epoch {self.current_epoch} - "
                f"train_acc: {acc:.2f}%, train_loss: {loss:.4f}"
            ))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        _, acc1 = forward_pass_classification(self.model, x, y)
        _, acc2 = forward_pass_classification(self.model, x, y, topk=2)

        self.log("top1_acc", acc1, on_epoch=True, prog_bar=True)
        self.log("top2_acc", acc2, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss, acc = forward_pass_classification(self.model, x, y)
        
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

class AutoEncoderModel(GeneralModel):
    def __init__(self, model, optimizer, scheduler):
        super().__init__(model, optimizer, scheduler)
        self.lp = self.model.lp

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = forward_pass_autoencoding(self.model, x, y, lp=self.lp)

        self.log(f"L{self.lp}_step", loss, on_epoch=False, prog_bar=True)
        self.log(f"L{self.lp}_epoch", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get(f"L{self.lp}_epoch")
        if loss is not None:
            print((
                f"Epoch {self.current_epoch} - "
                f"L{self.lp}_loss: {loss:.4f}"
            ))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = forward_pass_autoencoding(self.model, x, y, lp=self.lp)

        self.log(f"L{self.lp}_loss", loss, on_epoch=True, prog_bar=False)

class FlowModel(GeneralModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = forward_pass_flow(self.model, x, y)

        self.log("mse_step", loss, on_epoch=False, prog_bar=True)
        self.log("mse_epoch", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        loss = self.trainer.callback_metrics.get("mse_epoch")
        if loss is not None:
            print(f"Epoch {self.current_epoch} - mse_loss: {loss:.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = forward_pass_flow(self.model, x, y)

        self.log("mse_loss", loss, on_epoch=True, prog_bar=False)

class VAEModel(GeneralModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss_mse, loss_kl = forward_pass_var_aut(self.model, x, y)
        loss = loss_mse + loss_kl

        # normalize losses and log results
        loss_mse *= self.model.sigma
        loss_kl -= math.prod(self.model.z_shape)

        self.log("mse_step", loss_mse, on_epoch=False, prog_bar=True)
        self.log("mse_epoch", loss_mse, on_epoch=True, prog_bar=False)
        self.log("kl_step", loss_kl, on_epoch=False, prog_bar=True)
        self.log("kl_epoch", loss_kl, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        mse = self.trainer.callback_metrics.get("mse_epoch")
        kl = self.trainer.callback_metrics.get("kl_epoch")
        if mse is not None and kl is not None:
            print((
                f"Epoch {self.current_epoch} - "
                f"train_mse: {mse:.4f}, train_kl: {kl:.4f}"
            ))

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss_mse, loss_kl = forward_pass_var_aut(self.model, x, y)

        # normalize losses and log results
        loss_mse *= self.model.sigma
        loss_kl -= math.prod(self.model.z_shape)

        self.log("mse_error", loss_mse, on_epoch=True, prog_bar=True)
        self.log("kl_error", loss_kl, on_epoch=True, prog_bar=True)

class FlowClassifierModel(pl.LightningModule):
    """
    Uses the classifier to get the top 2 options, then uses
    the learned p(x|y) from the flow model to infer p(y|x)
    NOTE: this implementation assumes classes are balanced
    """
    def __init__(
            self, flow_nn, classifier, autoencoder, num_steps=100
        ):
        super().__init__()
        self.flow_nn = flow_nn
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.n_steps = num_steps

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # get (B, 2), indices
        _, y_hat = torch.topk(self.classifier(x), k=2, dim=1)
        # get (B, 2), probability of corresponding index
        z = self.autoencoder.encode(x)
        probs = flow_classify(self.flow_nn, z, y_hat, self.n_steps)
        # get (B, ), index of highest probability

        best_idx = torch.argmax(probs, dim=1)
        y_hat = y_hat[torch.arange(best_idx.shape[0]), best_idx]
        # y_hat = torch.where(
        #     probs[:, 0] > probs[:, 1], y_hat[:, 0], y_hat[:, 1]
        # )

        acc = 100 * torch.mean((y_hat == y).to(torch.float32))

        self.log("acc_epoch", acc, on_epoch=True, prog_bar=True)
