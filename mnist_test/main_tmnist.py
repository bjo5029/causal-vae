import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import os
import requests 

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

# ==========================================
# 2. 피처 추출 로직 (12개)
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
# 3. 데이터 다운로더 & 데이터셋
# ==========================================
def download_tmnist(filename="TMNIST_Data.csv"):
    url = "https://raw.githubusercontent.com/Ekinn7188/TMNIST-Classifier/master/TMNIST_Data.csv"
    
    print(f"Downloading TMNIST from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status() 
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("Download complete!")
    except Exception as e:
        print(f"다운로드 실패: {e}")
        print("다른 미러 링크를 시도하거나, Kaggle에서 'TMNIST_Data.csv'를 직접 다운로드해 주세요.")
        raise e

class MorphTMNIST(Dataset):
    def __init__(self, csv_file="TMNIST_Data.csv", train=True, limit_count=None):
        if not os.path.exists(csv_file):
            download_tmnist(csv_file)
            
        print(f"Loading TMNIST from {csv_file}...")
        self.df = pd.read_csv(csv_file)

        # Train/Test 분할
        total_len = len(self.df)
        split_idx = int(total_len * 0.8)
        
        if train:
            self.df = self.df.iloc[:split_idx]
            print(f"[Info] Training Set: {len(self.df)} images")
        else:
            self.df = self.df.iloc[split_idx:]
            print(f"[Info] Test Set: {len(self.df)} images")

        if limit_count is not None and limit_count < len(self.df):
            print(f"[Info] {limit_count}개만 사용하여 빠른 테스트를 진행합니다.")
            self.df = self.df.iloc[:limit_count]

        self.cached_data = []
        print(f"Pre-computing features...")
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            label = int(row['labels'])
            
            pixels = row.values[2:].astype(np.float32)
            img_np = pixels.reshape(28, 28) / 255.0
            img_tensor = torch.tensor(img_np).unsqueeze(0)
            
            m = extract_12_features(img_tensor)
            t_onehot = torch.zeros(10)
            t_onehot[label] = 1.0
            
            self.cached_data.append((img_tensor, m, t_onehot))
            
            if (i + 1) % 500 == 0:
                print(f"Processing... {i + 1}/{len(self.df)}", end='\r')
        
        print(f"\n[Done] Feature extraction complete!")

    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

# ==========================================
# 4. 모델 정의 (Causal Morph VAE)
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
    # 학습 데이터 로드 (limit_count로 개수 조절 가능)
    train_dataset = MorphTMNIST(train=True) 
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    model = CausalMorphVAE12().to(CONFIG["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LR"])
    
    print("\n[Start Training on TMNIST]...")
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0
        total_m_loss = 0
        
        for batch_idx, (x, m, t) in enumerate(train_loader):
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            recon_x, m_hat, mu, logvar = model(x, m, t)
            
            loss_recon = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            
            # Beta-VAE (Beta=2.0)
            loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * CONFIG["BETA"]
            
            # Morph Loss (가중치 200)
            loss_morph = F.mse_loss(m_hat, m, reduction='sum') * 200
            
            loss = loss_recon + loss_kld + loss_morph
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_m_loss += loss_morph.item()
            
        print(f"Epoch {epoch+1:02d} | Avg Loss: {total_loss/len(train_dataset):.2f} | M-Pred Loss: {total_m_loss/len(train_dataset):.4f}")
        
    return model

# ==========================================
# 6. 검증: Intervention (전체 수치 출력 수정됨)
# ==========================================
def run_intervention(model):
    print("\n[Running Intervention on TMNIST]...")
    model.eval()
    
    # 피처 이름 정의
    FEATURE_NAMES = [
        "01. Area        ", "02. Perimeter   ", "03. Thickness   ", "04. Major Axis  ", 
        "05. Eccentricity", "06. Orientation ", "07. Solidity    ", "08. Extent      ",
        "09. Aspect Ratio", "10. Euler Number", "11. H-Symmetry  ", "12. V-Symmetry  "
    ]
    
    # 테스트셋 로드
    test_dataset = MorphTMNIST(train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 1. 타겟 숫자 '1' 찾기
    target_digit = 1
    src_img, src_m, src_t = None, None, None
    
    for x, m, t in test_loader:
        if torch.argmax(t).item() == target_digit:
            src_img, src_m, src_t = x, m, t
            break
            
    src_img = src_img.to(CONFIG["DEVICE"])
    src_m = src_m.to(CONFIG["DEVICE"])
    src_t = src_t.to(CONFIG["DEVICE"])
    
    # 2. Z 추출 (고정된 스타일)
    with torch.no_grad():
        x_flat = src_img.view(1, -1)
        mu, logvar = model.encoder(torch.cat([x_flat, src_m, src_t], dim=1)).chunk(2, dim=1)
        z_fixed = mu 

    original_m_np = src_m.cpu().numpy()[0]
    
    print("-" * 60)
    print(f"Original Input Digit: {target_digit}")
    print("-" * 60)

    # 3. 0~9로 변환하며 모든 수치 비교 출력
    for i in range(10):
        t_fake = torch.zeros(1, 10).to(CONFIG["DEVICE"])
        t_fake[0, i] = 1.0
        
        with torch.no_grad():
            m_hat = model.morph_predictor(t_fake)
        
        pred_m_np = m_hat.cpu().numpy()[0]
        
        # (이름, 예측값, 원본값, 차이) 튜플 리스트
        changes = []
        for idx in range(12):
            val_pred = pred_m_np[idx]
            val_orig = original_m_np[idx]
            diff = abs(val_pred - val_orig)
            changes.append((FEATURE_NAMES[idx], val_pred, val_orig, diff))
        
        # 차이(diff) 기준 내림차순 정렬
        changes.sort(key=lambda x: x[3], reverse=True)
        
        # 0, 1, 8 같은 주요 변화 지점만 출력
        if i in [0, 1, 8, 2]: 
            print(f"\n>>> [Condition T={i}] Top Changes (vs Original '1')")
            print(f"{'Feature Name':<18} | {'Pred':<6} | {'Orig':<6} | {'Diff (Change)':<6}")
            print("-" * 50)
            for name, v_p, v_o, v_d in changes: # 전체 12개 다 출력
                print(f"{name:<18} | {v_p:.2f}   | {v_o:.2f}   | {v_d:.4f}")

    print("\nDone!")

def visualize_intervention_all(model, digit_to_test=1, save_dir="results"):
    print(f"\n[Visualizing Intervention] Converting digit '{digit_to_test}' to 0~9...")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    
    test_dataset = MorphTMNIST(train=False, limit_count=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    src_img, src_m, src_t = None, None, None
    
    for x, m, t in test_loader:
        if torch.argmax(t).item() == digit_to_test:
            src_img, src_m, src_t = x, m, t
            break 
            
    src_img = src_img.to(CONFIG["DEVICE"])
    src_m = src_m.to(CONFIG["DEVICE"])
    src_t = src_t.to(CONFIG["DEVICE"])
    
    with torch.no_grad():
        x_flat = src_img.view(1, -1)
        mu, logvar = model.encoder(torch.cat([x_flat, src_m, src_t], dim=1)).chunk(2, dim=1)
        z_fixed = mu 

    fig, axes = plt.subplots(1, 11, figsize=(20, 3))
    
    axes[0].imshow(src_img.cpu().squeeze(), cmap='gray')
    axes[0].set_title(f"Original {digit_to_test}\n(Source)", fontsize=10, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    for i in range(10):
        t_fake = torch.zeros(1, 10).to(CONFIG["DEVICE"])
        t_fake[0, i] = 1.0
        
        with torch.no_grad():
            m_hat = model.morph_predictor(t_fake)
            dec_input = torch.cat([m_hat, z_fixed], dim=1)
            gen_img = model.decoder(dec_input)
            
        ax = axes[i + 1]
        ax.imshow(gen_img.cpu().view(28, 28), cmap='gray')
        
        if i == digit_to_test:
            ax.set_title(f"Recon {i}\n(Self)", fontsize=10, color='red')
        else:
            ax.set_title(f"do(T={i})", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    file_name = f"intervention_tmnist_digit_{digit_to_test}.png"
    save_path = os.path.join(save_dir, file_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Done] Image saved successfully to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    trained_model = train_model()
    run_intervention(trained_model)
    
    visualize_intervention_all(trained_model, digit_to_test=0) 
    visualize_intervention_all(trained_model, digit_to_test=1)
    visualize_intervention_all(trained_model, digit_to_test=2)
    visualize_intervention_all(trained_model, digit_to_test=3)
    visualize_intervention_all(trained_model, digit_to_test=4)
    visualize_intervention_all(trained_model, digit_to_test=5)
    visualize_intervention_all(trained_model, digit_to_test=6)
    visualize_intervention_all(trained_model, digit_to_test=7) 
    visualize_intervention_all(trained_model, digit_to_test=8)
    visualize_intervention_all(trained_model, digit_to_test=9)
