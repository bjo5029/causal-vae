import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from config import CONFIG
from dataset import MorphMNIST12
from models import CausalMorphVAE12, SimpleClassifier
from train import train_model

def get_residuals(model, data_loader, device):
    """
    모든 데이터에 대해 (X - Recon_X) 잔차를 구해서 저장
    """
    model.eval()
    residuals = []
    labels = []
    
    print("[Residual Analysis] Generating residuals...")
    with torch.no_grad():
        for x, m, t in data_loader:
            x, m, t = x.to(device), m.to(device), t.to(device)
            x_flat = x.view(x.size(0), -1)
            
            # VAE Forward
            recon_x, _, _, _ = model(x, m, t)
            recon_x_img = recon_x.view(-1, 1, 28, 28)
            
            # Residual Calculation (Absolute Difference)
            # 정보가 남아있는지를 보려는 것이므로 부호보다는 크기가 중요할 수 있음
            # 하지만 원본 차이(Signed)를 그대로 주는 게 정보량이 더 많을 수 있으므로 그대로 사용
            res = x - recon_x_img
            
            residuals.append(res.cpu())
            # T is one-hot, convert to index
            labels.append(torch.argmax(t, dim=1).cpu())
            
    residuals = torch.cat(residuals, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return residuals, labels

def train_residual_classifier(residuals, labels, device, epochs=10):
    """
    잔차 이미지만 보고 원래 클래스(T)를 맞추는 분류기 학습
    """
    # 데이터셋 생성
    dataset = TensorDataset(residuals, labels)
    # Train/Test Split (8:2)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # 모델: SimpleClassifier (CNN)
    # 입력이 Residual 이미지(1x28x28)이므로 구조 동일하게 사용 가능
    classifier = SimpleClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    print(f"\n[Residual Analysis] Training classifier on residuals for {epochs} epochs...")
    
    for epoch in range(epochs):
        classifier.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            _, outputs = classifier(batch_x)
            loss = F.nll_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Test Accuracy check
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                _, outputs = classifier(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Test Accuracy: {acc:.2f}%")
        
    return acc

def main():
    # 0. 설정
    device = CONFIG["DEVICE"]
    print(f"Using device: {device}")
    
    # 1. VAE 모델 학습 (또는 로드)
    # 현재 체크포인트가 없으므로 학습 함수 호출
    print("Training VAE model first...")
    vae_model = train_model()
    
    # 2. 전체 데이터셋 로드
    full_dataset = MorphMNIST12(train=False, limit_count=None) # Test set 전체 사용
    data_loader = DataLoader(full_dataset, batch_size=128, shuffle=False)
    
    # 3. 잔차 데이터 생성
    residuals, labels = get_residuals(vae_model, data_loader, device)
    print(f"Collected {len(residuals)} residual samples.")
    
    # 4. 잔차 분류기 학습 및 평가
    final_acc = train_residual_classifier(residuals, labels, device)
    
    print("\n" + "="*50)
    print(f"FINAL RESULT: Residual Classification Accuracy = {final_acc:.2f}%")
    print("="*50)
    
    if final_acc < 20.0:
        print(">> PASS: Accuracy is low. Residuals contain little class info.")
    elif final_acc < 50.0:
        print(">> WARNING: Accuracy is moderate. Some class info might be leaking.")
    else:
        print(">> FAIL: Accuracy is high. Residuals contain significant class info.")
        print("   -> This implies M is insufficient or T->X direct path exists.")

if __name__ == "__main__":
    main()
