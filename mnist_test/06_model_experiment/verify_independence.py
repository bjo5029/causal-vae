import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MorphMNIST12 
import config

# Config
BATCH_SIZE = 128
EPOCHS = 10 
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineModel(nn.Module):
    """
    Model A: M -> X
    T 정보 없이 M만으로 X를 예측
    """
    def __init__(self, m_dim=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(m_dim, 64 * 7 * 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, m):
        h = self.fc(m)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)

class AugmentedModel(nn.Module):
    """
    Model B: M + T -> X
    M 외에 T 정보도 추가로 사용하여 X를 예측
    """
    def __init__(self, m_dim=12, t_dim=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(m_dim + t_dim, 64 * 7 * 7),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, m, t):
        x = torch.cat([m, t], dim=1)
        h = self.fc(x)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)

def train(model, loader, optimizer, use_t=False):
    model.train()
    total_loss = 0
    for x, m, t in loader:
        x, m, t = x.to(DEVICE), m.to(DEVICE), t.to(DEVICE)
        optimizer.zero_grad()
        
        if use_t:
            recon_x = model(m, t)
        else:
            recon_x = model(m)
            
        loss = nn.MSELoss()(recon_x, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, use_t=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, m, t in loader:
            x, m, t = x.to(DEVICE), m.to(DEVICE), t.to(DEVICE)
            if use_t:
                recon_x = model(m, t)
            else:
                recon_x = model(m)
            loss = nn.MSELoss()(recon_x, x) 
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    print("[Conditional Independence Test] Start...")
    
    # Data
    train_dataset = MorphMNIST12(train=True)
    test_dataset = MorphMNIST12(train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model A: M -> X
    print("\n[Model A] Training Baseline (M -> X)...")
    model_a = BaselineModel().to(DEVICE)
    opt_a = optim.Adam(model_a.parameters(), LR)
    for epoch in range(EPOCHS):
        loss = train(model_a, train_loader, opt_a, use_t=False)
        print(f"Epoch {epoch+1}: Loss {loss:.6f}")
    mse_a = evaluate(model_a, test_loader, use_t=False)
    print(f" -> Test MSE (Model A): {mse_a:.6f}")
    
    # Model B: M + T -> X
    print("\n[Model B] Training Augmented (M + T -> X)...")
    model_b = AugmentedModel().to(DEVICE)
    opt_b = optim.Adam(model_b.parameters(), LR)
    for epoch in range(EPOCHS):
        loss = train(model_b, train_loader, opt_b, use_t=True)
        print(f"Epoch {epoch+1}: Loss {loss:.6f}")
    mse_b = evaluate(model_b, test_loader, use_t=True)
    print(f" -> Test MSE (Model B): {mse_b:.6f}")
    
    # Conclusion
    print("\n[Result Analysis]")
    print(f"Baseline MSE (M->X): {mse_a:.6f}")
    print(f"Augmented MSE (M+T->X): {mse_b:.6f}")
    print(f"Difference: {abs(mse_a - mse_b):.6f}")
    
    if mse_b < mse_a * 0.95:
        print(">> T provides significant information relating to X given M. (Independence rejected)")
    else:
        print(">> Performance is similar. X and T are conditionally independent given M. (Supported)")
