import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CONFIG
from dataset import MorphMNIST12
from models import CausalMorphVAE12, SimpleClassifier, LatentDiscriminator

def train_model():
    """VAE 모델 + 적대적 학습 루프"""
    train_dataset = MorphMNIST12(train=True) 
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    # Models
    vae = CausalMorphVAE12().to(CONFIG["DEVICE"])
    discriminator = LatentDiscriminator().to(CONFIG["DEVICE"])
    
    # Optimizers
    opt_vae = optim.Adam(vae.parameters(), lr=CONFIG["LR"])
    opt_d = optim.Adam(discriminator.parameters(), lr=CONFIG["LR"])
    
    print("\n[Start Adversarial Training]...")
    for epoch in range(CONFIG["EPOCHS"]):
        vae.train()
        discriminator.train()
        
        total_loss = 0
        total_m_loss = 0
        total_d_loss = 0
        total_adv_loss = 0
        
        for batch_idx, (x, m, t) in enumerate(train_loader):
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            t_indices = torch.argmax(t, dim=1)
            
            # -------------------------------------
            # 1. Train Discriminator (D)
            # -------------------------------------
            opt_d.zero_grad()
            
            with torch.no_grad():
                # VAE Forward (Only Encode)
                # z 필요, D 학습에는 recon_x가 필요하지 않음
                # 단순하게 전체 forward를 사용하고 detach 처리
                _, _, mu, logvar = vae(x, m, t)
                z = vae.reparameterize(mu, logvar).detach()
                _, _, mu, logvar = vae(x, m, t)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            
            # D tries to classify T from Z
            d_logits = discriminator(z)
            loss_d = F.cross_entropy(d_logits, t_indices)
            
            loss_d.backward()
            opt_d.step()
            total_d_loss += loss_d.item()
            
            # -------------------------------------
            # 2. Train VAE (Generator)
            # -------------------------------------
            opt_vae.zero_grad()
            
            recon_x, m_hat, mu, logvar = vae(x, m, t)
            
            # Standard VAE Losses
            loss_recon = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
            kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kld = kld_element * CONFIG["BETA"]
            loss_morph = F.mse_loss(m_hat, m, reduction='sum') * 100
            
            # Confusion Loss
            # VAE는 Discriminator가 z에서 t를 예측하지 못하도록(균등 분포 출력) 유도
            # KL(Uniform || P)를 최소화하여 P가 균등 분포에 가깝게 만듦 (엔트로피 최대화)
            z_sample = vae.reparameterize(mu, logvar)
            d_logits_fake = discriminator(z_sample)
            
            # 균등 분포 타겟(1/10) 및 로그 확률 계산
            target_uniform = torch.full_like(d_logits_fake, 1.0 / CONFIG["T_DIM"])
            log_probs = F.log_softmax(d_logits_fake, dim=1)
            
            loss_adv = F.kl_div(log_probs, target_uniform, reduction='batchmean') * CONFIG["LAMBDA_ADV"] * 100 
            # Reconstruction Loss와 스케일을 맞추기 위해 100을 곱함
            loss = loss_recon + loss_kld + loss_morph + loss_adv
            loss.backward()
            opt_vae.step()
            
            total_loss += loss.item()
            total_m_loss += loss_morph.item()
            total_adv_loss += loss_adv.item()
            
        print(f"Epoch {epoch+1:02d} | Avg Loss: {total_loss/len(train_dataset):.1f} | M-Loss: {total_m_loss/len(train_dataset):.2f} | D-Loss: {total_d_loss/len(train_loader):.4f} | Adv-Loss: {total_adv_loss/len(train_dataset):.2f}")
        
    import os
    os.makedirs("results", exist_ok=True)
    save_path = "results/model_final.pt"
    torch.save(vae.state_dict(), save_path)
    print(f"\n[Train] Model saved to {save_path}")
    
    return vae

def train_external_classifier(device):
    """외부 분류기 학습 루프"""
    print("\n[External Encoder] Training a simple classifier for evaluation...")
    model = SimpleClassifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # 분류기는 일반 MNIST로 학습 (Fake data로 하는 거 아님)
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(10): 
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            _, output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print(f" -> Classifier Epoch {epoch+1} done.")
    
    print("[External Encoder] Ready")
    return model
