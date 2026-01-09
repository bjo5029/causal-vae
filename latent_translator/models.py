# models.py
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
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTVAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        latent_dim=512,
        img_size=(384, 640),
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

        # 1. Stem (5단계 Downsampling -> Stride 32)
        # 384->12, 640->20
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(),   # /2
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(),           # /4
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(),         # /8
            nn.Conv2d(128, embed_dim, 3, 2, 1), nn.BatchNorm2d(embed_dim), nn.LeakyReLU(), # /16
            nn.Conv2d(embed_dim, embed_dim, 3, 2, 1), nn.BatchNorm2d(embed_dim), nn.LeakyReLU(), # /32
        )

        self.grid_h = self.img_height // 32
        self.grid_w = self.img_width // 32
        self.num_patches = self.grid_h * self.grid_w

        # 2. Transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = nn.Sequential(*[ViTBlock(embed_dim, heads, mlp_dim) for _ in range(depth)])

        self.to_latent = nn.LayerNorm(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_var = nn.Linear(embed_dim, latent_dim)

        # 3. Decoder
        self.decoder_input = nn.Linear(latent_dim, embed_dim * self.grid_h * self.grid_w)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 3, 2, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(), ResBlock(128), # x2
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(), ResBlock(64),          # x4
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(), ResBlock(32),           # x8
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(), ResBlock(16),           # x16
            nn.ConvTranspose2d(16, 16, 3, 2, 1, 1), nn.BatchNorm2d(16), nn.LeakyReLU(),                         # x32
            nn.Conv2d(16, in_channels, 3, 1, 1),
        )

    def encode(self, x):
        x = self.stem(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # Pos Embedding 크기가 안 맞으면 자동으로 자르거나 보간해서 더함 (Safety)
        if x.shape[1] != self.pos_embedding.shape[1]:
             # 임시 처리: 그냥 앞부분만 잘라서 씀 (어차피 main.py에서 보간해서 넣어줌)
             x += self.pos_embedding[:, :x.shape[1]]
        else:
             x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        cls_out = self.to_latent(x[:, 0])
        return self.fc_mu(cls_out), self.fc_var(cls_out)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.embed_dim, self.grid_h, self.grid_w)
        return self.decoder(result)

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return recons, input, mu, log_var
    