from math import prod

import torch
from torch import nn, Tensor


class SparseConv2d(nn.Module):
    """
    Sparse convolutional layer. Spacial convolution is done
    in parallel for each channel using the "groups" argument.

    see: "Mobilenetv2: Inverted residuals and linear bottlenecks"
    """
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                dim2, dim2, kernel_size=3, padding=1,
                groups=dim2, padding_mode="reflect"
            ),
            nn.GELU(),
            nn.Conv2d(dim2, dim1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, dim1, h, w)"""
        return x + self.network(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int):
        """
        Self attention for images; input is 4D (batch, channels, h, w)

        Args:
            dim: Feature dimension at each spatial position (n_channels)
            n_heads: Number of attention heads.
            head_dim: Dimension of attention embedding (query/key/value)
        """
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        attn_dim = n_heads * head_dim

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, d, h, w = x.shape
        x_flat = x.view(B, d, h * w).transpose(1, 2)  # (B, h*w, d)

        # (B, h*w, attn_dim)
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # (B, n_heads, h*w, head_dim)
        q = q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # (B, n_heads, h*w, h*w)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        attended = torch.matmul(attn, v)  # (B, n_heads, h*w, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, h*w, -1)
        attended = self.out_proj(attended)  # (B, h*w, d)
        attended = attended.transpose(1, 2).view(B, d, h, w)  # (B, d, h, w)

        out = x + attended

        return out


class Conformer(nn.Module):
    """Wrapper for SparseConv2d and SelfAttention"""
    def __init__(self, dim: int, mult: int, n_heads: int, head_dim: int):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, head_dim)
        self.conv = SparseConv2d(dim, mult * dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x)
        x = self.conv(x)
        return x


class AutoEncoder(nn.Module):
    """Auto encoder architecture for MNIST images"""
    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 48,
        mult: int = 5,
        n_layers: int = 1,
        n_heads: int = 3,
        head_dim: int = 8,
    ):
        super().__init__()
        self.init_args = dict(locals())
        del self.init_args['self']
        del self.init_args['__class__']

        self.z_shape: tuple[int, ...] = (1, 7, 7)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim1, kernel_size=5, stride=2, padding=2),  # 14
            nn.GELU(),
            nn.Conv2d(
                dim1, dim2, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 7
            *(Conformer(dim2, mult, n_heads, head_dim) for _ in range(n_layers)),
            nn.Conv2d(dim2, 1, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, dim2, kernel_size=1),
            *(Conformer(dim2, mult, n_heads, head_dim) for _ in range(n_layers)),
            nn.Upsample(scale_factor=2),            # -> 14
            nn.Conv2d(
                dim2, dim1, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dim1, 1, kernel_size=5, padding=2),
        )

    def encode(self, x: Tensor) -> Tensor:
        """x: (batch, 1, 28, 28)"""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """z: (batch, *z_shape)"""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class VarAutoEncoder(nn.Module):
    """
    Class for autoencoder that learns an encoding of the MNIST dataset.
    """
    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 48,
        mult: int = 5,
        n_layers: int = 1,
        n_heads: int = 3,
        head_dim: int = 8,
    ):
        super().__init__()
        self.init_args = dict(locals())
        del self.init_args['self']
        del self.init_args['__class__']

        self.z_shape: tuple[int, ...] = (1, 6, 6)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim1, kernel_size=5, stride=2),  # 12
            nn.GELU(),
            nn.Conv2d(
                dim1, dim2, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 6
            *(Conformer(dim2, mult, n_heads, head_dim) for _ in range(n_layers)),
            nn.Conv2d(dim2, 2, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, dim2, kernel_size=1),
            *(Conformer(dim2, mult, n_heads, head_dim) for _ in range(n_layers)),
            nn.Upsample(scale_factor=2),            # -> 12x12
            nn.Conv2d(
                dim2, dim1, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.Upsample(28),
            nn.Conv2d(dim1, 1, kernel_size=5, padding=2),
        )

    def get_encoding(self, x: Tensor) -> tuple[Tensor, Tensor]:
        repr = self.encoder(x)  # (batch, 2, ...)
        x_mean = repr[:, :1]
        log_var = repr[:, 1:]
        return x_mean, torch.exp(log_var)

    def get_decoding(self, z: Tensor) -> tuple[Tensor, Tensor]:
        pred_x = self.decoder(z)
        return pred_x, torch.ones_like(pred_x)

    def encode(self, x: Tensor, random: bool | float = False) -> Tensor:
        z_mean, z_var = self.get_encoding(x)
        ep = random * torch.normal(torch.zeros_like(z_mean), torch.ones_like(z_mean))
        return z_mean + torch.sqrt(z_var) * ep

    def decode(self, z: Tensor) -> Tensor:  # no randomness
        pred_x, _ = self.get_decoding(z)
        return pred_x

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class Diffusion(nn.Module):
    """
    Learns to generate images directly.
    Must implement z_dim and num_classes.
    """
    def __init__(
        self,
        dim: int = 32,
        mult: int = 4,
        n_layers: int = 1,
        n_heads: int = 3,
        head_dim: int = 8,
        n_classes: int = 47,
        z_shape: tuple[int, ...] = (1, 6, 6),
    ):
        super().__init__()
        self.init_args = dict(locals())
        del self.init_args['self']
        del self.init_args['__class__']

        self.z_shape = z_shape
        self.n_classes = n_classes

        self.network = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=5, padding=2),
            nn.GELU(),
            *(Conformer(dim, mult, n_heads=n_heads, head_dim=head_dim) for _ in range(n_layers)),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=5, padding=2),
        )
        self.yemb = nn.Linear(n_classes, prod(z_shape), bias=False)
        self.temb = nn.Linear(n_classes, prod(z_shape))
        self.nums = nn.Parameter(
            torch.linspace(0, 1, n_classes),
            requires_grad=False,
        )
        self.log_scalar = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, xt: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """xt: (B, *z_shape),  t: (B, 1, ..., 1),  y: (B, n_classes)"""
        t = torch.exp((t.view(-1, 1) - self.nums)**2 / torch.exp(self.log_scalar))
        t = self.temb(t).view(-1, *self.z_shape)

        y = self.yemb(y).view(-1, *self.z_shape)
        x = torch.cat((xt, t, y), dim=1)

        out = self.network(x)
        return out
