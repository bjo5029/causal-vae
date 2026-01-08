import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import os

# ==========================================
# 1. 설정 (Configuration)
# ==========================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CONFIG = {
    "BATCH_SIZE": 128,
    "EPOCHS": 30,
    "LR": 1e-3,
    "Z_DIM": 10,
    "M_DIM": 12,
    "T_DIM": 10,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SEED": 42,
    "BETA": 2.0
}

torch.manual_seed(CONFIG["SEED"])

# 피처 이름 
FEATURE_NAMES = [
    "Area", "Perimeter", "Thickness", "MajorAxis", 
    "Eccentricity", "Orientation", "Solidity", "Extent",
    "AspectRatio", "Euler", "H_Symmetry", "V_Symmetry"
]

# ==========================================
# 2. 피처 추출 로직
# ==========================================
def extract_12_features(img_tensor):
    img = img_tensor.squeeze().numpy()
    binary = img > 0.2
    
    if np.sum(binary) == 0: return torch.zeros(12, dtype=torch.float32)

    labeled_img = label(binary)
    props = regionprops(labeled_img)
    if not props: return torch.zeros(12, dtype=torch.float32)
    prop = props[np.argmax([p.area for p in props])]

    f1_area = prop.area / 784.0
    f2_perimeter = prop.perimeter / 100.0
    dist_map = distance_transform_edt(binary)
    f3_thickness = np.max(dist_map) / 5.0
    f4_major_axis = prop.major_axis_length / 28.0
    f5_eccentricity = prop.eccentricity
    f6_orientation = (prop.orientation + np.pi/2) / np.pi
    f7_solidity = prop.solidity
    f8_extent = prop.extent
    minr, minc, maxr, maxc = prop.bbox
    height = maxr - minr
    width = maxc - minc
    f9_aspect_ratio = (width / height) if height > 0 else 0
    f9_aspect_ratio = np.clip(f9_aspect_ratio, 0, 3.0) / 3.0 
    f10_euler = (prop.euler_number + 2) / 4.0 
    flipped_lr = np.fliplr(img)
    f11_h_symmetry = 1.0 - np.mean(np.abs(img - flipped_lr))
    flipped_ud = np.flipud(img)
    f12_v_symmetry = 1.0 - np.mean(np.abs(img - flipped_ud))

    features = [f1_area, f2_perimeter, f3_thickness, f4_major_axis, f5_eccentricity, 
                f6_orientation, f7_solidity, f8_extent, f9_aspect_ratio, 
                f10_euler, f11_h_symmetry, f12_v_symmetry]
    return torch.tensor(features, dtype=torch.float32)

# ==========================================
# 3. 데이터셋 정의 
# ==========================================
class MorphMNIST12(Dataset):
    def __init__(self, train=True, limit_count=None): 
        self.mnist = datasets.MNIST(root='../data', train=train, download=True, 
                                    transform=transforms.ToTensor())
        
        if limit_count is not None and limit_count < len(self.mnist):
            print(f"[Info] 전체 데이터 중 {limit_count}개만 사용")
            self.indices = range(limit_count)
        else:
            print(f"[Info] 전체 데이터({len(self.mnist)}개) 사용")
            self.indices = range(len(self.mnist))

        self.cached_data = []
        print(f"Pre-computing features for {'Train' if train else 'Test'} set...")
        
        for i, idx in enumerate(self.indices):
            img, label = self.mnist[idx]
            m = extract_12_features(img)
            t_onehot = torch.zeros(10)
            t_onehot[label] = 1.0
            self.cached_data.append((img, m, t_onehot))
            if (i + 1) % 500 == 0:
                print(f"Processing... {i + 1}/{len(self.indices)}", end='\r')
        
        print(f"\nFeature extraction complete")

    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

# ==========================================
# 4. 모델 정의 
# ==========================================
class CausalMorphVAE12(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_dim = CONFIG["M_DIM"]
        self.t_dim = CONFIG["T_DIM"]
        self.z_dim = CONFIG["Z_DIM"]
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + self.m_dim + self.t_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.z_dim * 2) 
        )
        self.morph_predictor = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.m_dim)
        )
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
        enc_input = torch.cat([x_flat, m, t], dim=1)
        mu, logvar = self.encoder(enc_input).chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        
        m_hat = self.morph_predictor(t)
        dec_input = torch.cat([m_hat, z], dim=1)
        recon_x = self.decoder(dec_input)
        
        return recon_x, m_hat, mu, logvar

# ==========================================
# 5. 학습 루프
# ==========================================
def train_model():
    train_dataset = MorphMNIST12(train=True) 
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    model = CausalMorphVAE12().to(CONFIG["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    print("\n[Start Training]...")
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        total_m_loss = 0
        
        for batch_idx, (x, m, t) in enumerate(train_loader):
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            recon_x, m_hat, mu, logvar = model(x, m, t)
            
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

# ==========================================
# 6. 10x10 조합 CSV 저장 함수
# ==========================================
def export_intervention_csv_10x10(model, save_path="causal_intervention_10x10.csv"):
    print(f"\n[Exporting CSV] Collecting representative samples for digits 0-9")
    model.eval()
    
    # 1. 테스트셋 전체 로드
    test_dataset = MorphMNIST12(train=False, limit_count=None) 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 숫자 0~9별 대표 샘플 1개씩 수집
    digit_samples = {}
    
    for x, m, t in test_loader:
        label = torch.argmax(t).item()
        
        # 아직 수집 안 된 숫자라면 저장
        if label not in digit_samples:
            digit_samples[label] = (x, m, t)
            print(f" -> Found representative image for digit: {label}")
        
        # 0~9 다 모았으면 중단
        if len(digit_samples) == 10:
            break
            
    print(f"Found samples for all 10 digits. Calculating 10x10 intervention...")
    
    results = []
    
    # 3. 10(Source) x 10(Target) Loop
    with torch.no_grad():
        # Source Loop: 0, 1, ..., 9
        for source_digit in range(10):
            # 대표 이미지 가져오기
            x, m, t = digit_samples[source_digit]
            x = x.to(CONFIG["DEVICE"])
            m = m.to(CONFIG["DEVICE"])
            t = t.to(CONFIG["DEVICE"])
            
            # 원본 형태값 (NumPy)
            original_m_np = m.cpu().numpy()[0]
            
            # (1) 스타일(Z) 추출 (고정)
            x_flat = x.view(x.size(0), -1)
            mu, _ = model.encoder(torch.cat([x_flat, m, t], dim=1)).chunk(2, dim=1)
            z_fixed = mu
            
            # Target Loop: 0, 1, ..., 9 (Intervention)
            for target_digit in range(10):
                # 가상의 조건 T 생성
                t_fake = torch.zeros(1, 10).to(CONFIG["DEVICE"])
                t_fake[0, target_digit] = 1.0
                
                # (2) 형태(M) 예측
                m_hat = model.morph_predictor(t_fake)
                pred_m_np = m_hat.cpu().numpy()[0]
                
                # (3) 데이터 기록
                row = {
                    "Source_Digit": source_digit,
                    "Target_Digit": target_digit
                }
                
                # 12개 피처 각각 기록
                for i, feat_name in enumerate(FEATURE_NAMES):
                    val_orig = original_m_np[i]
                    val_pred = pred_m_np[i]
                    val_diff = val_pred - val_orig # 변화량 (부호 포함)
                    # 절대값 차이를 원하면 abs() 씌우면 됨. 여기선 방향성 보려고 부호 유지함.
                    
                    row[f"{feat_name}_Orig"] = val_orig
                    row[f"{feat_name}_Pred"] = val_pred
                    row[f"{feat_name}_Diff"] = val_diff
                
                results.append(row)

    # 4. CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"\n[Done] Successfully saved {len(df)} rows to '{save_path}'")
    print(" (Row 1~10: Source 0 -> Target 0...9)")
    print(" (Row 11~20: Source 1 -> Target 0...9)")

# ==========================================
# 7. 메인 실행
# ==========================================
if __name__ == "__main__":
    trained_model = train_model()
    
    # 10x10 매트릭스 CSV 추출 실행
    export_intervention_csv_10x10(trained_model, save_path="intervention_matrix_10x10.csv")