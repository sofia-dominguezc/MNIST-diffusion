import torch
from torch import nn

class MBConv2d(nn.Module):
    """
    (B, d1, h, w) -> (B, d2, h, w) -> (B, d3, h, w)
    if linear=False, then we must have d1==d3
    """
    def __init__(self, d1, d2, d3, linear=True):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(d1, d2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                d2, d2, kernel_size=3, padding=1,
                groups=d2, padding_mode="reflect"
            ),
            nn.GELU(),
            nn.Conv2d(d2, d3, kernel_size=1),
        )
        if linear:
            self.linear = nn.Conv2d(d1, d3, kernel_size=1)

    def forward(self, x):
        if hasattr(self, "linear"):
            return self.linear(x) + self.network(x)
        return x + self.network(x)

class SelfAttention(nn.Module):
    def __init__(self, d, n_heads, head_dim):
        """
        Args:
            d (int): Feature dimension at each spatial position.
            n_heads (int): Number of attention heads.
            head_dim (int): Dimension of attention embedding (query/key/value)
        """
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = head_dim
        attn_dim = n_heads * head_dim

        self.q_proj = nn.Linear(d, attn_dim)
        self.k_proj = nn.Linear(d, attn_dim)
        self.v_proj = nn.Linear(d, attn_dim)
        self.out_proj = nn.Linear(attn_dim, d)

    def forward(self, x):
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

        scores = torch.matmul(q, k.transpose(-2, -1)) \
                    / (self.head_dim ** 0.5)  # (B, n_heads, h*w, h*w)
        attn = torch.softmax(scores, dim=-1)

        attended = torch.matmul(attn, v)  # (B, n_heads, h*w, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, h*w, -1)
        attended = self.out_proj(attended)  # (B, h*w, d)
        attended = attended.transpose(1, 2).view(B, d, h, w)  # (B, d, h, w)

        out = x + attended

        return out

class Conformer(nn.Module):
    """Does self attention followed by an MBConv layer.
    It's like a transformer layer, but with spacial convolution
    in the MLP part."""
    def __init__(self, d, k, n_heads, head_dim):
        super().__init__()
        self.attn = SelfAttention(d, n_heads, head_dim)
        self.conv = MBConv2d(d, k*d, d, linear=False)

    def forward(self, x):
        x = self.attn(x)
        x = self.conv(x)
        return x

class AutoEncoder(nn.Module):
    """
    Class for autoencoder that learns an encoding of the MNIST dataset.
    shape of encoding: (B, 2, 7, 7)
    """
    def __init__(
            self, d1=32, d2=32, k=5, n=1, n_heads=3, head_dim=8, p_norm=1
        ):
        super().__init__()
        self.z_shape = (1, 7, 7)
        self.up = nn.Upsample(scale_factor=2)
        self.ac = nn.GELU()
        self.lp = p_norm  # lp norm to minimize in training
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d1, kernel_size=5, padding=2, stride=2),  
            self.ac,
            nn.Conv2d(d1, d2, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 7x7
            *[Conformer(d2, k, n_heads, head_dim) for _ in range(n)],
            nn.Conv2d(d2, 1, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, d2, kernel_size=3, padding=1),
            *[Conformer(d2, k, n_heads, head_dim) for _ in range(n)],
            self.up,                    # -> 14x14
            nn.Conv2d(d2, d1, kernel_size=3, padding=1),
            self.ac,
            self.up,                            
            nn.Conv2d(d1, 1, kernel_size=5, padding=2),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

class VarAutoEncoder(nn.Module):
    """
    Class for autoencoder that learns an encoding of the MNIST dataset.
    shape of encoding: (B, 2, 7, 7)
    """
    def __init__(
            self, d1=24, d2=32, k=5, n=1, n_heads=4, head_dim=8, sigma=0.2
        ):
        super().__init__()
        self.z_shape = (1, 6, 6)
        self.ac = nn.GELU()
        # define models
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d1, kernel_size=5, stride=2),  # 12
            self.ac,
            nn.Conv2d(
                d1, d2, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            self.ac,
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 6
            *[Conformer(d2, k, n_heads, head_dim) for _ in range(n)],
            nn.Conv2d(d2, 2, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, d2, kernel_size=1),
            *[Conformer(d2, k, n_heads, head_dim) for _ in range(n)],
            nn.Upsample(scale_factor=2),            # -> 12x12
            nn.Conv2d(
                d2, d1, kernel_size=3, padding=1, padding_mode='reflect'
            ),
            nn.Upsample(28),
            nn.Conv2d(d1, 1, kernel_size=5, padding=2),
        )
        self.sigma = nn.Parameter(
            torch.tensor(sigma, dtype=torch.float32),
            requires_grad=False
        )

    def get_encoding(self, x):
        repr = self.encoder(x)
        mu = repr[:, :1]
        log_var = repr[:, 1:]
        return mu, torch.exp(log_var) / 100

    def get_decoding(self, z):
        x_hat = self.decoder(z)
        return x_hat, self.sigma * torch.ones_like(x_hat)

    def encode(self, x, random=False):
        mu, var = self.get_encoding(x)
        ep = random * torch.normal(torch.zeros_like(mu), torch.ones_like(mu)).view(mu.shape)
        return mu + torch.sqrt(var) * ep

    def decode(self, z, random=False):  # no randomness
        x_hat, var = self.get_decoding(z)
        return x_hat

    def forward(self, x):
        return self.decode(self.encode(x))

class Classifier(nn.Module):
    """Works on original images (1, 28, 28)"""
    def __init__(
        self, d1=48, d2=64, k=6, n=2, n_heads=6, head_dim=8, n_class=10
    ):
        super().__init__()
        self.num_classes = n_class
        self.ac = nn.GELU()
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.network = nn.Sequential(
            nn.Conv2d(1, d1, kernel_size=5, padding=2, stride=2),  # 14
            self.ac,
            self.max,                         # 7
            nn.Conv2d(d1, d2, kernel_size=1),
            nn.Conv2d(
                d2, d2, kernel_size=3, padding=1,
                groups=d2, padding_mode='reflect',
            ),
            self.ac,
            self.max,                         # 3
            *[Conformer(d2, k, n_heads=n_heads, head_dim=head_dim) \
                for _ in range(n)],
            self.ac,
            self.max,
            nn.Flatten(),
            nn.Linear(d2, n_class),
        )

    def forward(self, x):
        """x: (B, 1, 28, 28). out: (B, num_classes)"""
        return self.network(x)

class FlowMatching(nn.Module):
    """
    Learns to generate images directly.
    Must implement z_dim and num_classes.
    """
    def __init__(self,
        d=24, k=4, n=1, n_heads=4, head_dim=16,
        n_class=10, p=0.1, z_dim=7,
    ):
        super().__init__()
        self.z_shape = (1, z_dim, z_dim)
        self.num_classes = n_class
        self.p = p  # probability of deleting label when training
        self.network = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=5, padding=2),
            nn.GELU(),
            *[Conformer(d, k, n_heads=n_heads, head_dim=head_dim) \
                for _ in range(n)],
            nn.GELU(),
            nn.Conv2d(d, 1, kernel_size=5, padding=2),
        )
        self.yemb = nn.Linear(n_class, z_dim**2, bias=False)
        self.temb = nn.Linear(n_class, z_dim**2)
        self.nums = nn.Parameter(
            torch.arange(1, n_class+1, dtype=torch.float32) \
                * torch.pi / (2 * n_class), requires_grad=False,
        )

    def forward(self, x, t, y):
        """x: (B, *z_shape),  t: (B, 1, ...),  y: (B, n_class)"""
        t = torch.cos(t.view(-1, 1) * self.nums)
        t = self.temb(t).view(-1, *self.z_shape)
        y = self.yemb(y).view(-1, *self.z_shape)
        x = torch.cat((x, t, y), dim=1)

        out = self.network(x)
        return out
