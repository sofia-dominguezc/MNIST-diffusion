import torch
from torch import nn, Tensor


class Diffusion(nn.Module): ...


class DiffusionViT(Diffusion):
    """Vision Transformer with patch embedding and positional encoding"""
    def __init__(
        self,
        in_channels: int = 1,
        image_size: tuple[int, int] = (28, 28),
        patch_size: int = 7,
        dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        mlp_ratio: float = 4.0,
        n_classes: int = 10,
    ):
        super().__init__()

        self.init_args = dict(locals())
        del self.init_args['self']
        del self.init_args['__class__']
        self.n_classes = n_classes
        self.dim = dim
        self.n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim))

        self.transformer = nn.Sequential(*(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=int(dim*mlp_ratio),
                batch_first=True,
                activation="gelu",
            ) for _ in range(n_layers)
        ))

        self.unpatch = nn.ConvTranspose2d(
            dim, in_channels, kernel_size=patch_size, stride=patch_size
        )
        self.y_encoder = nn.Embedding(n_classes + 1, dim * 2)

    def _sinusoidal_embedding(self, t: Tensor, dim: int) -> Tensor:
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """x: (B, in_channels, image_size, image_size), t: (B, 1)"""
        x = self.patch_embed(x)  # (B, hidden_dim, n_patches_h, n_patches_w)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, D)

        t_emb = self._sinusoidal_embedding(      # (B, 2, D)
            t.flatten(), dim=self.dim * 2,
        ).view(B, 2, D)
        y = torch.where(y == -1, self.n_classes, y)  # -1 -> last embedding
        y_emb = self.y_encoder(y).view(B, 2, D)  # (B, 2, D)

        x = x + self.pos_embed
        x = torch.cat([x, t_emb, y_emb], dim=1)  # (B, n_pathces + 4, D)
        x = self.transformer(x)
        x = x[:, :-4, :]                         # (B, n_patches, D)

        x = x.transpose(1, 2).view(B, D, H, W)
        x = self.unpatch(x)  # (B, in_channels, image_size, image_size)
        return x


class DiffusionCNN(Diffusion):
    """Vision Transformer with patch embedding and positional encoding"""
    def __init__(
        self,
        in_channels: int = 1,
        image_size: tuple[int, int] = (28, 28),
        patch_size: int = 7,
        dim: int = 64,
        n_layers: int = 4,
        mlp_ratio: float = 4.0,
        n_classes: int = 10,
    ):
        super().__init__()

        self.init_args = dict(locals())
        del self.init_args['self']
        del self.init_args['__class__']
        self.n_classes = n_classes
        self.dim = dim
        self.n_layers = n_layers
        self.n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim))

        hidden_dim = int(dim * mlp_ratio)
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1),
                nn.Dropout(0.05),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                nn.Conv2d(hidden_dim, dim, kernel_size=1),
            ) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

        self.unpatch = nn.ConvTranspose2d(
            dim, in_channels, kernel_size=patch_size, stride=patch_size
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * n_layers * 2),
        )
        self.y_encoder = nn.Embedding(n_classes + 1, dim * n_layers * 2)

    def _sinusoidal_embedding(self, t: Tensor, dim: int) -> Tensor:
        """Generate sinusoidal time embeddings"""
        device = t.device
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """x: (B, in_channels, image_size, image_size), t: (B, 1)"""
        x = self.patch_embed(x)  # (B, hidden_dim, n_patches_h, n_patches_w)

        t = self._sinusoidal_embedding(t.flatten(), dim=self.hidden_dim)
        t_embeddings = self.time_mlp(t).chunk(self.n_layers, dim=-1)
        y = torch.where(y == -1, self.n_classes, y)  # -1 -> last embedding
        y_embeddings = self.y_encoder(y).chunk(self.n_layers, dim=-1)

        for layer, norm, t_emb, y_emb in zip(self.layers, self.layer_norms, t_embeddings, y_embeddings):
            shift, scale = (t_emb + y_emb).chunk(2, dim=-1)  # (B, D)
            shift, scale = shift[:, :, None, None], scale[:, :, None, None]
            x = shift + (1 + scale) * x
            x = x + layer(norm(x))

        x = self.unpatch(x)  # (B, in_channels, image_size, image_size)
        return x


if __name__ == "__main__":
    model = DiffusionViT()
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e3:.1f}K params")
    data = torch.randn(5, 1, 28, 28)
    out = model(data, t=torch.zeros(data.shape[0]), y=torch.zeros(data.shape[0], 10))
    print(out.shape)
