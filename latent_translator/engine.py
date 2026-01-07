import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

def train_vit_vae(model, loader: DataLoader, optimizer, device: str, epochs: int, beta: float = 1.0) -> None:
    """
    ViTVAE 모델 학습을 위한 루프
    Return: (recons, input, mu, log_var)
    """
    model.train()
    
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_samples = 0
        
        for batch in loader:
            x = batch["x"].to(device)
            optimizer.zero_grad()
            
            # ViTVAE의 Forward 리턴값은 4개
            recons, _, mu, log_var = model(x)
            
            # Loss 계산
            recon_loss = F.mse_loss(recons, x, reduction="mean")
            kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + beta * kld_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        avg_loss = total_loss / max(n_samples, 1)
        print(f"[ViTVAE] Epoch {ep:03d}/{epochs} | Loss: {avg_loss:.6f}")

@torch.no_grad()
def extract_vit_latents(model, loader: DataLoader, device: str) -> np.ndarray:
    """
    ViTVAE 모델에서 Latent Vector(mu) 추출
    """
    model.eval()
    zs = []
    # print("Extracting latents...") # 로그 많으면 주석 처리하기
    for batch in loader:
        x = batch["x"].to(device)
        # ViTVAE는 encode() 메서드를 사용
        mu, _ = model.encode(x)
        zs.append(mu.detach().cpu().numpy())
    
    return np.concatenate(zs, axis=0)
