import torch
import torch.nn.functional as F
from tqdm import tqdm

def loss_function(recon_x, x, m_hat, m, mu, logvar, gamma=20.0):
    # 1. Image Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. Mechanism Loss (T->M)
    m_loss = F.mse_loss(m_hat, m, reduction='sum')
    
    # 3. KLD
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total
    loss = recon_loss + (gamma * m_loss) + kld
    return loss, recon_loss, m_loss

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    # tqdm progress bar
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        x, m, t = batch
        x, m, t = x.to(device), m.to(device), t.to(device)
        
        optimizer.zero_grad()
        recon_x, m_hat, mu, logvar = model(x, m, t)
        
        loss, l_recon, l_m = loss_function(recon_x, x, m_hat, m, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'M_Loss': f"{l_m.item():.2f}"})
        
    return total_loss / len(loader.dataset)
