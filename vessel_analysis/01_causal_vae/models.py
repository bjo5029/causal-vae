import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoImageProcessor, AutoModel
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
            
            # 1. 6x10 -> 12x20 (Scale 2)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            
            # 2. 12x20 -> 24x40 -> 48x80 -> 96x160 -> 192x320 -> 384x640 -> 768x1280
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()
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

class PhikonLoss(nn.Module):
    """
    Phikon Perceptual Loss (Histology ViT) with Random Cropping
    """
    def __init__(self):
        super().__init__()
        # Load Phikon (owkin/phikon)
        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        self.model = AutoModel.from_pretrained("owkin/phikon")
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Normalization (ImageNet stats as per Phikon docs)
        self.register_buffer("mean", torch.tensor(self.processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.processor.image_std).view(1, 3, 1, 1))
        
        self.crop_size = 224

    def forward(self, input, target):
        # Input/Target: [B, 1, H, W]
        B, C, H, W = input.shape
        
        # 1. Random Crop (Same coords for both)
        # Handle edge case where H or W < 224 (Unlikely given 768x1280)
        top = torch.randint(0, H - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, W - self.crop_size + 1, (1,)).item()
        
        input_crop = input[:, :, top:top+self.crop_size, left:left+self.crop_size]
        target_crop = target[:, :, top:top+self.crop_size, left:left+self.crop_size]
        
        # 2. To 3 Channels (Repeat)
        x = input_crop.repeat(1, 3, 1, 1)
        y = target_crop.repeat(1, 3, 1, 1)
        
        # 3. Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # 4. Extract Features (spatial patches)
        # Phikon output: last_hidden_state [B, 197, 768] (CLS + 14x14 patches)
        # Use patches (1:) to preserve spatial texture info. 
        # CLS (0) is too global and causes "blobby" artifacts.
        x_out = self.model(x).last_hidden_state[:, 1:, :]
        y_out = self.model(y).last_hidden_state[:, 1:, :]
        
        # 5. MSE Loss
        return F.mse_loss(x_out, y_out, reduction='sum')


# ==========================================
# Causal ViT VAE (Wrapper for Pretrained)
# ==========================================

try:
    from vit_backbone import ViTVAE
except ImportError:
    print("Warning: Could not import ViTVAE from vit_backbone.py.")
    ViTVAE = object

class CausalViTVAE(nn.Module):
    """
    Wrapper to use Pretrained ViTVAE as the backbone for Causal VAE.
    Original ViTVAE: X -> Z -> X (Reconstruction only)
    Causal ViTVAE:   (X, M, T) -> Z -> (Z, M) -> X
    """
    def __init__(self, pretrained_path=None):
        super().__init__()
        
        # 1. Load Backbone
        # Initialize with same params as refs/model.py usually expects
        # Adjust these if the pretrained model used different args
        self.backbone = ViTVAE(
            img_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"]),
            patch_size=32,
            embed_dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            latent_dim=512 # Matched to Pretrained Weights
        )
        
        if pretrained_path:
            print(f"[CausalViTVAE] Loading backbone weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=CONFIG["DEVICE"])
            self.backbone.load_state_dict(state_dict, strict=False)
    
            # Optional: Freeze Backbone
            # for param in self.backbone.parameters():
            #     param.requires_grad = False
            # print("[CausalViTVAE] Frozen backbone weights: Only training adapters & morph predictor.")
        
        # 2. Dimensions
        self.vit_embed_dim = 256
        self.vit_latent_dim = 512 # Matched to Pretrained Weights
        self.my_z_dim = CONFIG["Z_DIM"]
        self.m_dim = CONFIG["M_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        
        # 3. Adapters
        
        # Encoder Adapter:
        # ViT Stem+Transformer gives us a feature vector (CLS token) of size `embed_dim`
        # We need to combine this with M and T to predict OUR Z.
        self.enc_adapter = nn.Sequential(
            nn.Linear(self.vit_embed_dim + self.m_dim + self.t_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.my_z_dim * 2) # mu, logvar
        )
        
        # Decoder Adapter:
        # We have (Z, M). ViT Decoder expects `latent_dim` (128).
        # We map (Z, M) -> ViT Latent
        self.dec_adapter = nn.Sequential(
            nn.Linear(self.my_z_dim + self.m_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.vit_latent_dim)
        )
        
        # Morph Predictor (Same as before, not part of ViT)
        self.morph_predictor_shared = nn.Sequential(
            nn.Linear(self.t_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2)
        )
        self.morph_predictor_mu = nn.Linear(64, self.m_dim)
        self.morph_predictor_logvar = nn.Linear(64, self.m_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m, t):
        # --- 1. Encoder (Backbone + Adapter) ---
        # Use ViT Stem + Transformer
        # Re-using ViTVAE code pieces manually to access CLS token
        
        # Step A: Stem
        features = self.backbone.stem(x)
        
        # Step B: Flatten & Pos Emb
        from einops import rearrange, repeat
        features = rearrange(features, "b c h w -> b (h w) c")
        b, n, _ = features.shape
        cls_tokens = repeat(self.backbone.cls_token, "1 1 d -> b 1 d", b=b)
        features = torch.cat((cls_tokens, features), dim=1)
        features += self.backbone.pos_embedding[:, : (n + 1)]
        features = self.backbone.dropout(features)
        
        # Step C: Transformer
        features = self.backbone.transformer(features)
        
        # Step D: Get CLS Token (Representation of Image)
        cls_out = self.backbone.to_latent(features[:, 0]) # (B, 256)
        
        # Step E: Adapter (Condition on M, T)
        enc_input = torch.cat([cls_out, m, t], dim=1)
        mu, logvar = self.enc_adapter(enc_input).chunk(2, dim=1)
        
        # Safety Clamping
        logvar = torch.clamp(logvar, min=-10, max=10)
        mu = torch.clamp(mu, min=-100, max=100)
        
        z = self.reparameterize(mu, logvar)
        
        # --- 2. Morph Predictor ---
        h = self.morph_predictor_shared(t)
        m_mu = self.morph_predictor_mu(h)
        m_logvar = self.morph_predictor_logvar(h)
        m_logvar = torch.clamp(m_logvar, min=-10, max=10)
        m_hat = m_mu
        
        # --- 3. Decoder (Adapter + Backbone) ---
        # Combine Real M and Z
        dec_input_our = torch.cat([m, z], dim=1)
        
        # Map to ViT Latent Space
        z_vit = self.dec_adapter(dec_input_our) # (B, 128)
        
        # Use ViT Decoder
        recon_x = self.backbone.decode(z_vit)
        
        return recon_x, m_hat, mu, logvar, m_mu, m_logvar
