import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalBioVAE(nn.Module):
    def __init__(self, img_channels=1, m_dim=12, t_dim=19, latent_dim=64):
        super().__init__()
        self.m_dim = m_dim
        self.t_dim = t_dim
        
        # [Encoder] q(Z | X, M, T)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            # 이미지 크기가 유동적이므로 Global Pooling 사용
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten()
        )
        # 256채널 * 4 * 4 = 4096
        self.flatten_dim = 256 * 4 * 4 
        
        self.enc_fc = nn.Sequential(
            nn.Linear(self.flatten_dim + m_dim + t_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # [Decoder Stage 1] T -> M
        self.mechanism_net = nn.Sequential(
            nn.Linear(t_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, m_dim) 
        )

        # [Decoder Stage 2] [Z, M_hat] -> X
        self.dec_input = nn.Linear(latent_dim + m_dim, self.flatten_dim)
        
        # Upsampling (4x4 -> 8x8 -> ... -> Target Size)
        # Adaptive하게 하려면 복잡하므로, 여기서는 resize된 크기에 맞춰서 키움
        # ConvTranspose2d는 고정된 배수만큼 키우므로, dataset에서 
        # 32의 배수(예: 384, 640)로 넣는 거 중요
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # x2
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # x4
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # x8
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),    # x16
        )
        # 4x4에서 시작해서 4번 2배씩 키우면 -> 64x64
        # 원래 크기로 맞추기 위해 forward에서 interpolate 사용

    def encode(self, x, m, t_onehot):
        x_feat = self.enc_conv(x)
        combined = torch.cat([x_feat, m, t_onehot], dim=1)
        h = self.enc_fc(combined)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, m, t):
        t_onehot = F.one_hot(t, num_classes=self.t_dim).float()
        
        mu, logvar = self.encode(x, m, t_onehot)
        z = self.reparameterize(mu, logvar)
        
        # Stage 1
        m_hat = self.mechanism_net(t_onehot)
        
        # Stage 2
        z_m_input = torch.cat([z, m_hat], dim=1)
        x_feat = self.dec_input(z_m_input)
        x_feat = x_feat.view(-1, 256, 4, 4) # (B, 256, 4, 4)
        
        out = self.dec_conv(x_feat) # (B, 1, 64, 64) 정도 크기
        
        # 최종적으로 입력 이미지 크기에 맞춰서 늘려줌 (Bilinear Interpolation)
        recon_x = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return recon_x, m_hat, mu, logvar
    