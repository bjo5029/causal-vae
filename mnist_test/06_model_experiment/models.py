import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

class CausalMorphVAE12(nn.Module):
    """
    제안하는 Causal Morph-VAE 모델 (CNN Version)
    구조: T -> M -> X (Condition -> Morphology -> Image)
    """
    def __init__(self):
        super().__init__()
        self.m_dim = CONFIG["M_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        self.z_dim = CONFIG["Z_DIM"]
        
        # [Encoder] (X) -> Features -> + (M, T) -> Z
        # Input: (1, 28, 28)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), # (32, 14, 14)
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), # (64, 7, 7)
            nn.Flatten()
        )
        self.enc_flat_dim = 64 * 7 * 7
        
        self.enc_fc = nn.Sequential(
            nn.Linear(self.enc_flat_dim + self.m_dim + self.t_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.z_dim * 2) 
        )
        
        # [Morph Predictor] T -> M' (Causal Logic)
        # 확률 분포 P(M|T) = N(mu, sigma^2) 예측
        self.morph_predictor_shared = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.ReLU()
        )
        self.morph_predictor_mu = nn.Linear(128, self.m_dim)
        self.morph_predictor_logvar = nn.Linear(128, self.m_dim)
        
        # [Decoder] (M', Z) -> X
        self.dec_fc = nn.Sequential(
            nn.Linear(self.m_dim + self.z_dim, self.enc_flat_dim),
            nn.ReLU()
        )
        
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), # (32, 14, 14)
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid() # (1, 28, 28)
        )

    def morph_predictor(self, t):
        """Helper for visualization (returns mean only)"""
        h = self.morph_predictor_shared(t)
        return self.morph_predictor_mu(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m, t):
        # 1. Encode
        # x: (B, 1, 28, 28)
        x_feat = self.enc_conv(x)
        enc_input = torch.cat([x_feat, m, t], dim=1)
        mu, logvar = self.enc_fc(enc_input).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        
        # 2. Predict Morphology (Probabilistic)
        h = self.morph_predictor_shared(t)
        m_mu = self.morph_predictor_mu(h)
        m_logvar = self.morph_predictor_logvar(h)

        m_hat = m_mu # 기본 출력은 평균값 사용 (Loss 계산용)
        
        # 3. Decode
        # Decoder 학습 시에는 "Real M"을 사용함.
        # 이렇게 해야 Decoder가 P(X|M, Z)를 정확히 학습하며, Predictor의 오차에 의존하지 않음.
        dec_input = torch.cat([m, z], dim=1) # m_hat -> m (Real M) 변경
        h = self.dec_fc(dec_input)
        h = h.view(-1, 64, 7, 7)
        recon_x = self.dec_conv(h)
        
        return recon_x, m_hat, mu, logvar, m_mu, m_logvar

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

class LatentDiscriminator(nn.Module):
    """
    적대적 학습을 위한 Discriminator
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
    