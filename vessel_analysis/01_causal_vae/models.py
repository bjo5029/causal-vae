import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

class CausalVesselVAE(nn.Module):
    """
    Causal Morph-VAE for Vessel Images (768x1280)
    Structure: T -> M -> X
    Input: (1, 768, 1280)
    Latent: Z_DIM (64)
    """
    def __init__(self):
        super().__init__()
        self.m_dim = CONFIG["M_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        self.z_dim = CONFIG["Z_DIM"]
        
        # [Encoder]
        # Input: (1, 768, 1280)
        # Layer 1: (32, 384, 640)
        # Layer 2: (64, 192, 320)
        # Layer 3: (128, 96, 160)
        # Layer 4: (256, 48, 80)
        # Layer 5: (512, 24, 40)
        # Layer 6: (512, 12, 20)
        # Layer 7: (512, 6, 10) -> Flatten
        
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2), # L1
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), # L2
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), # L3
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2), # L4
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), # L5
            nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), # L6
            nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), # L7
            nn.Flatten()
        )
        
        # Final Feature Map: 6 x 10 (H/128 x W/128)
        self.enc_flat_dim = 512 * 6 * 10
        self.enc_fc = nn.Sequential(
            nn.Linear(self.enc_flat_dim + self.m_dim + self.t_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.z_dim * 2) 
        )
        
        # [Morph Predictor] T -> M (Gaussian P(M|T))
        self.morph_predictor_shared = nn.Sequential(
            nn.Linear(self.t_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2)
        )
        self.morph_predictor_mu = nn.Linear(64, self.m_dim)
        self.morph_predictor_logvar = nn.Linear(64, self.m_dim)
        
        # [Decoder] (M, Z) -> X
        self.dec_fc = nn.Sequential(
            nn.Linear(self.m_dim + self.z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.enc_flat_dim),
            nn.ReLU()
        )
        
        self.dec_conv = nn.Sequential(
            # Input: (512, 14, 26)
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), # L7 -> (512, 28, 52)
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), # L6 -> (512, 56, 104)
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), # L5 -> (512, 112, 208) - Wait, channels were 256 here in encoder
            # Correction: Encoder L5 output is 512, Input is 256. 
            # Decoder L5 should output 256.
            # Sequence: L7(512->512), L6(512->512), L5(512->256), L4(256->128), L3(128->64), L2(64->32), L1(32->1)
            
            # Re-checking Encoder:
            # L5: 256->512. L6: 512->512. L7: 512->512.
            # Decoder:
            # L7t: 512->512. (Out: 28x52)
            # L6t: 512->512. (Out: 56x104)
            # L5t: 512->256. (Out: 112x208)
            # L4t: 256->128. (Out: 224x416)
            # L3t: 128->64.  (Out: 448x832)
            # L2t: 64->32.   (Out: 896x1664)
            # L1t: 32->1.    (Out: 1792x3328)
            
            # Implementation:
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), # L5t (Wait, indexes are confusing. Just stack)
             
            # 28x52 -> 56x104
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), 
            # 56x104 -> 112x208
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            # 112x208 -> 224x416
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            # 224x416 -> 448x832
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            # 448x832 -> 896x1664
            # 896x1664 -> 1792x3328
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid() 
        )
        
        # Wait, I might have messed up the layer count in the Sequential above. Let's be explicit.
        self.dec_conv = nn.Sequential(
            # Z -> (512, 6, 10)
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), # -> (512, 28, 52)
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), # -> (512, 56, 104)
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), # -> (256, 112, 208)
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # -> (128, 224, 416)
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # -> (64, 448, 832)
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),   # -> (32, 896, 1664)
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()                      # -> (1, 1792, 3328)
        )


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m, t):
        # 1. Encode
        x_feat = self.enc_conv(x)
        enc_input = torch.cat([x_feat, m, t], dim=1)
        mu, logvar = self.enc_fc(enc_input).chunk(2, dim=1)
        # Safety Clamping for Z
        logvar = torch.clamp(logvar, min=-10, max=10)
        mu = torch.clamp(mu, min=-100, max=100) # Optional safety for mu as well
        z = self.reparameterize(mu, logvar)
        
        # 2. Predict Morphology (Probabilistic)
        h = self.morph_predictor_shared(t)
        m_mu = self.morph_predictor_mu(h)
        m_logvar = self.morph_predictor_logvar(h)
        m_logvar = torch.clamp(m_logvar, min=-10, max=10) # Safety Clamp
        
        m_hat = m_mu # For reconstruction loss, use Mean
        
        # 3. Decode uses REAL M during training
        dec_input = torch.cat([m, z], dim=1) 
        h_dec = self.dec_fc(dec_input)
        h_dec = h_dec.view(-1, 512, 6, 10) # Reshape to spatial
        recon_x = self.dec_conv(h_dec)
        
        return recon_x, m_hat, mu, logvar, m_mu, m_logvar

class LatentDiscriminator(nn.Module):
    """
    Discriminator for Adversarial Training (matching Z prior)
    """
    def __init__(self):
        super().__init__()
        self.z_dim = CONFIG["Z_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.t_dim) # Logits
        )
        
    def forward(self, z):
        return self.net(z)
