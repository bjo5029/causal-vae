import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CONFIG
from dataset import MorphMNIST12
from models import CausalMorphVAE12, SimpleClassifier

def train_model():
    """VAE 모델 학습 루프"""
    train_dataset = MorphMNIST12(train=True) 
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    model = CausalMorphVAE12().to(CONFIG["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    print("\n[Start Training VAE]...")
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        total_m_loss = 0
        
        for batch_idx, (x, m, t) in enumerate(train_loader):
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            recon_x, m_hat, mu, logvar = model(x, m, t)
            
            # Loss 계산
            loss_recon = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kld = kld_element * CONFIG["BETA"]
            loss_morph = F.mse_loss(m_hat, m, reduction='sum') * 100
            
            loss = loss_recon + loss_kld + loss_morph
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_m_loss += loss_morph.item()
            
        print(f"Epoch {epoch+1:02d} | Avg Loss: {total_loss/len(train_dataset):.2f} | M-Pred Loss: {total_m_loss/len(train_dataset):.4f}")
        
    return model

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
