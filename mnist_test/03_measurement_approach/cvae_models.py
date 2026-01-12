
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

class ConditionalVAE(nn.Module):
    """
    Conditional VAE for T -> X generation.
    M is NOT used in the generative process.
    Encoder: q(z|x, t)
    Decoder: p(x|z, t)
    """
    def __init__(self):
        super().__init__()
        self.z_dim = CONFIG["Z_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        
        # Encoder
        # Input: X (1 channel) + T (t_dim channels broadcasted) -> No, simpler to concatenate flattened.
        # Let's stick to CNN features + T concatenation at FC layer for better spatial handling?
        # Or simple MLP for MNIST? The previous model was CNN, let's keep CNN for X but fuse T.
        
        # Encoder Conv Part
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), # 3x3
            nn.ReLU(),
        )
        
        # Encoder FC Part
        # Flattened conv output: 64 * 3 * 3 = 576
        # We concatenate T here.
        self.enc_fc_mu = nn.Linear(576 + self.t_dim, self.z_dim)
        self.enc_fc_logvar = nn.Linear(576 + self.t_dim, self.z_dim)
        
        # Decoder FC Part
        # Input: Z + T
        self.dec_fc = nn.Linear(self.z_dim + self.t_dim, 64 * 7 * 7)
        
        # Decoder Conv Part
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), # 28x28
            nn.Sigmoid() 
        )

    def encode(self, x, t):
        # x: (B, 1, 28, 28)
        # t: (B, 10)
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1) # Flatten -> (B, 576)
        
        # Concatenate x features and t
        h_t = torch.cat([h, t], dim=1) # (B, 576 + 10)
        
        mu = self.enc_fc_mu(h_t)
        logvar = self.enc_fc_logvar(h_t)
        return mu, logvar

    def decode(self, z, t):
        # z: (B, z_dim)
        # t: (B, t_dim)
        z_t = torch.cat([z, t], dim=1) # (B, z_dim + t_dim)
        
        h = self.dec_fc(z_t)
        h = h.view(-1, 64, 7, 7)
        
        recon_x = self.dec_conv(h)
        return recon_x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t):
        mu, logvar = self.encode(x, t)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, t)
        return recon_x, mu, logvar
