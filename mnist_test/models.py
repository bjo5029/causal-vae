import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

class CausalMorphVAE12(nn.Module):
    """
    제안하는 Causal Morph-VAE 모델
    구조: T -> M -> X (Condition -> Morphology -> Image)
    """
    def __init__(self):
        super().__init__()
        self.m_dim = CONFIG["M_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        self.z_dim = CONFIG["Z_DIM"]
        
        # Encoder: (X, M, T) -> Z
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + self.m_dim + self.t_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.z_dim * 2) 
        )
        
        # Morph Predictor: T -> M' (Causal Logic)
        self.morph_predictor = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.m_dim)
        )
        
        # Decoder: (M', Z) -> X
        self.decoder = nn.Sequential(
            nn.Linear(self.m_dim + self.z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m, t):
        x_flat = x.view(x.size(0), -1)
        
        # 1. Encode
        enc_input = torch.cat([x_flat, m, t], dim=1)
        mu, logvar = self.encoder(enc_input).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        
        # 2. Predict Morphology
        m_hat = self.morph_predictor(t)
        
        # 3. Decode
        dec_input = torch.cat([m_hat, z], dim=1)
        recon_x = self.decoder(dec_input)
        
        return recon_x, m_hat, mu, logvar

class SimpleClassifier(nn.Module):
    """
    검증용 External Classifier
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        feature = F.relu(self.fc1(x)) # t-SNE용 특징 벡터
        out = self.fc2(feature)
        return feature, F.log_softmax(out, dim=1)
    