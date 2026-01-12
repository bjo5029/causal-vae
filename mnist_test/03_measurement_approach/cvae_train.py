
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import MorphMNIST12
from cvae_models import ConditionalVAE

def train_cvae():
    device = CONFIG["DEVICE"]
    
    # Dataset (Uses same dataset but we ignore 'm' in training)
    train_dataset = MorphMNIST12(train=True) 
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    model = ConditionalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    print("\n[Start CVAE Training (T->X)]...")
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, (x, _, t) in enumerate(train_loader):
            # Ignore m
            x, t = x.to(device), t.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, t)
            
            # Loss: BCE + KLD
            # Flatten for BCE
            loss_recon = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
            
            # KLD
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # Use Beta=1.0 for CVAE usually, or small beta to prioritize reconstruction quality
            loss_kld = kld_element * 1.0 # Standard VAE
            
            loss = loss_recon + loss_kld
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kld += loss_kld.item()
            
        avg_loss = total_loss / len(train_dataset)
        avg_recon = total_recon / len(train_dataset)
        avg_kld = total_kld / len(train_dataset)
        
        print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.1f} (Recon: {avg_recon:.1f}, KLD: {avg_kld:.1f})")
        
    return model
