import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv(x)


class ViTBlock(nn.Module):
    """Standard Transformer Encoder Block"""

    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-Attention
        qkv = self.norm1(x)
        attn_out, _ = self.attn(qkv, qkv, qkv)
        x = x + attn_out
        # Feed Forward
        x = x + self.mlp(self.norm2(x))
        return x


class ViTVAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        latent_dim=128,
        img_size=(768, 1280),
        patch_size=32,
        embed_dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
    ):
        super(ViTVAE, self).__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size

        # ========================
        # 1. Hybrid Stem (CNN Feature Extractor)
        # ========================
        # Patch Size 32만큼 Downsampling 수행 (Stride 2 * 2 * 2 * 2 * 2 = 32)
        # H: 768 -> 24, W: 1280 -> 40
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),  # /16
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # /32
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(),
        )

        # Stem 이후의 Grid Size 계산
        self.grid_h = self.img_height // 32  # 24
        self.grid_w = self.img_width // 32  # 40
        self.num_patches = self.grid_h * self.grid_w

        # ========================
        # 2. Transformer Encoder
        # ========================
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = nn.Sequential(
            *[ViTBlock(embed_dim, heads, mlp_dim) for _ in range(depth)]
        )

        # Latent Projection
        self.to_latent = nn.LayerNorm(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_var = nn.Linear(embed_dim, latent_dim)

        # ========================
        # 3. Decoder (CNN Based)
        # ========================
        # Latent -> Initial Spatial Map
        self.decoder_input = nn.Linear(
            latent_dim, embed_dim * self.grid_h * self.grid_w
        )

        # Upsampling Layers (Mirroring the Stem)
        self.decoder = nn.Sequential(
            # Input: (B, embed_dim, 24, 40)
            nn.ConvTranspose2d(
                embed_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # x2 -> 48, 80
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            ResBlock(128),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # x4 -> 96, 160
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            ResBlock(64),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # x8 -> 192, 320
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            ResBlock(32),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # x16 -> 384, 640
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                16, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # x32 -> 768, 1280
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            # nn.Sigmoid() # MSE Loss를 위해 Sigmoid 제거 (기존 코드 유지)
        )

    def encode(self, x):
        # 1. Stem (CNN)
        x = self.stem(x)  # (B, embed_dim, H/32, W/32)

        # 2. Flatten & Add Position Embedding
        # (B, C, H, W) -> (B, H*W, C)
        x = rearrange(x, "b c h w -> b (h w) c")
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        # 3. Transformer
        x = self.transformer(x)

        # 4. Latent (Use CLS token)
        cls_out = self.to_latent(x[:, 0])
        mu = self.fc_mu(cls_out)
        log_var = self.fc_var(cls_out)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Project back to spatial dimensions
        result = self.decoder_input(z)
        result = result.view(-1, self.embed_dim, self.grid_h, self.grid_w)

        # CNN Upsampling
        result = self.decoder(result)
        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return recons, input, mu, log_var
