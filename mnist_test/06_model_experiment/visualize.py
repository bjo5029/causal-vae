import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from config import CONFIG, FEATURE_NAMES
from dataset import MorphMNIST12

def export_intervention_csv_10x10(model, digit_samples, save_path="intervention_10x10.csv"):
    """[기능 1] Intervention 결과를 CSV로 저장 (수치 분석용)"""
    print(f"\n[CSV Export] Calculating 10x10 intervention...")
    results = []
    model.eval()
    
    with torch.no_grad():
        for source_digit in range(10):
            if source_digit not in digit_samples: continue
                
            x, m, t = digit_samples[source_digit]
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            original_m_np = m.cpu().numpy()[0]
            
            x_feat = model.enc_conv(x)
            enc_input = torch.cat([x_feat, m, t], dim=1)
            mu, _ = model.enc_fc(enc_input).chunk(2, dim=1)
            
            for target_digit in range(10):
                t_fake = torch.zeros(1, 10).to(CONFIG["DEVICE"])
                t_fake[0, target_digit] = 1.0
                
                m_hat = model.morph_predictor(t_fake)
                pred_m_np = m_hat.cpu().numpy()[0]
                
                row = {"Source_Digit": source_digit, "Target_Digit": target_digit}
                
                for i, feat_name in enumerate(FEATURE_NAMES):
                    val_orig = original_m_np[i]
                    val_pred = pred_m_np[i]
                    val_diff = val_pred - val_orig 
                    row[f"{feat_name}_Orig"] = val_orig
                    row[f"{feat_name}_Pred"] = val_pred
                    row[f"{feat_name}_Diff"] = val_diff
                
                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"[Done] CSV saved to '{save_path}'")

def visualize_intervention_grid_with_original(model, digit_samples, save_path="intervention_grid.png"):
    """[기능 2] 원본 이미지와 Intervention 결과를 그리드로 시각화"""
    print(f"\n[Image Gen] Generating 10x11 intervention grid (Col 1: Original)...")
    model.eval()
    fig, axes = plt.subplots(10, 11, figsize=(22, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    with torch.no_grad():
        for row_idx, source_digit in enumerate(range(10)):
            if source_digit not in digit_samples: continue

            x, m, t = digit_samples[source_digit]
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            
            x_feat = model.enc_conv(x)
            enc_input = torch.cat([x_feat, m, t], dim=1)
            mu, _ = model.enc_fc(enc_input).chunk(2, dim=1)
            z_fixed = mu
            
            # [Column 0] 원본
            ax_orig = axes[row_idx, 0]
            ax_orig.imshow(x.cpu().view(28, 28), cmap='gray')
            ax_orig.axis('off')
            if row_idx == 0:
                ax_orig.set_title("Original", fontsize=12, fontweight='bold', color='blue')
            ax_orig.set_ylabel(f"Src {source_digit}", fontsize=12, fontweight='bold')
            
            # [Column 1~10] Intervention
            for i, target_digit in enumerate(range(10)):
                t_fake = torch.zeros(1, 10).to(CONFIG["DEVICE"])
                t_fake[0, target_digit] = 1.0
                
                m_hat = model.morph_predictor(t_fake)
                dec_input = torch.cat([m_hat, z_fixed], dim=1)
                h = model.dec_fc(dec_input)
                h = h.view(-1, 64, 7, 7)
                recon_x = model.dec_conv(h)
                
                ax = axes[row_idx, i+1]
                ax.imshow(recon_x.cpu().view(28, 28), cmap='gray')
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(f"Tgt {target_digit}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Done] Image saved to '{save_path}'")

def visualize_z_clustering(model, save_path="z_tsne_plot.png"):
    """[기능 3] Latent Vector Z 분포 확인"""
    print(f"\n[Validation] Extracting Z vectors for t-SNE visualization...")
    model.eval()
    
    test_dataset = MorphMNIST12(train=False, limit_count=2000)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    z_list = []
    label_list = []
    
    with torch.no_grad():
        for x, m, t in test_loader:
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            x_feat = model.enc_conv(x)
            enc_input = torch.cat([x_feat, m, t], dim=1)
            mu, _ = model.enc_fc(enc_input).chunk(2, dim=1)
            z_list.append(mu.cpu().numpy())
            label_list.append(torch.argmax(t, dim=1).cpu().numpy())
            
    z_all = np.concatenate(z_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    
    print(f"[t-SNE] Running t-SNE on {len(z_all)} vectors...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_embedded = tsne.fit_transform(z_all)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], 
                          c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10), label='Digit Class (0-9)')
    plt.title("t-SNE visualization of Latent Vector Z", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Done] t-SNE plot saved to '{save_path}'")
    plt.show()

def verify_visualization(model):
    """[기능 4] PCA와 다양한 Perplexity로 t-SNE 검증"""
    print(f"\n[Verification] Running PCA & Multi-Perplexity t-SNE...")
    model.eval()
    
    test_dataset = MorphMNIST12(train=False, limit_count=2000)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    z_list = []
    label_list = []
    
    with torch.no_grad():
        for x, m, t in test_loader:
            x, m, t = x.to(CONFIG["DEVICE"]), m.to(CONFIG["DEVICE"]), t.to(CONFIG["DEVICE"])
            x_feat = model.enc_conv(x)
            enc_input = torch.cat([x_feat, m, t], dim=1)
            mu, _ = model.enc_fc(enc_input).chunk(2, dim=1)
            z_list.append(mu.cpu().numpy())
            label_list.append(torch.argmax(t, dim=1).cpu().numpy())
            
    z_all = np.concatenate(z_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_all)
    scatter = axes[0].scatter(z_pca[:, 0], z_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    axes[0].set_title("Method 1: PCA", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE (Low Perplexity)
    tsne_low = TSNE(n_components=2, perplexity=5, random_state=42, init='pca', learning_rate='auto')
    z_tsne_low = tsne_low.fit_transform(z_all)
    axes[1].scatter(z_tsne_low[:, 0], z_tsne_low[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    axes[1].set_title("Method 2: t-SNE (Perp=5)", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    # t-SNE (High Perplexity)
    tsne_high = TSNE(n_components=2, perplexity=50, random_state=42, init='pca', learning_rate='auto')
    z_tsne_high = tsne_high.fit_transform(z_all)
    axes[2].scatter(z_tsne_high[:, 0], z_tsne_high[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    axes[2].set_title("Method 3: t-SNE (Perp=50)", fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes, ticks=range(10), label='Digit Class')
    plt.savefig("verification_plots.png", dpi=300)
    print("[Done] Verification plots saved.")
    plt.show()

def validate_and_analyze_outliers(vae_model, classifier, device):
    """[기능 5] Real vs Fake 클러스터링 및 이상치 분석"""
    print("\n[Validation] Generating Data for t-SNE & Outlier Analysis...")
    vae_model.eval()
    classifier.eval()
    
    test_dataset = MorphMNIST12(train=False, limit_count=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    digit_counts = {i: 0 for i in range(10)}
    collected_real = [] 
    
    for x, m, t in test_loader:
        label = torch.argmax(t).item()
        if digit_counts[label] < 20:
            digit_counts[label] += 1
            collected_real.append((x, m, t, label))
        if all(c >= 20 for c in digit_counts.values()):
            break
            
    print(f" -> Collected 200 Real images. Generating Fakes...")
    
    features_list = []
    labels_list = []
    markers_list = []
    images_list = []
    
    with torch.no_grad():
        for (x, m, t, src_label) in collected_real:
            x, m, t = x.to(device), m.to(device), t.to(device)
            
            # (A) Real Image
            real_feat, _ = classifier(x)
            features_list.append(real_feat.cpu().numpy())
            labels_list.append(src_label)
            markers_list.append('Real')
            images_list.append(x.cpu())
            
            # (B) Fake Image
            x_feat = vae_model.enc_conv(x)
            enc_input = torch.cat([x_feat, m, t], dim=1)
            mu, _ = vae_model.enc_fc(enc_input).chunk(2, dim=1)
            z_fixed = mu
            
            for target_digit in range(10):
                t_fake = torch.zeros(1, 10).to(device)
                t_fake[0, target_digit] = 1.0
                
                m_hat = vae_model.morph_predictor(t_fake)
                dec_input = torch.cat([m_hat, z_fixed], dim=1)
                h = vae_model.dec_fc(dec_input)
                h = h.view(-1, 64, 7, 7)
                recon_x = vae_model.dec_conv(h)
                recon_x_img = recon_x.view(1, 1, 28, 28)
                
                fake_feat, _ = classifier(recon_x_img)
                features_list.append(fake_feat.cpu().numpy())
                labels_list.append(target_digit)
                markers_list.append('Fake')
                images_list.append(recon_x_img.cpu())

    X_all = np.concatenate(features_list, axis=0)
    y_all = np.array(labels_list)
    m_all = np.array(markers_list)
    
    print(f" -> Running t-SNE on {len(X_all)} vectors...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_all)
    
    # Plotting Logic (Labeled t-SNE)
    plt.figure(figsize=(14, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        idx_fake = (y_all == digit) & (m_all == 'Fake')
        plt.scatter(X_embedded[idx_fake, 0], X_embedded[idx_fake, 1], 
                    marker='^', s=30, alpha=0.3, color=colors[digit])
        
        idx_real = (y_all == digit) & (m_all == 'Real')
        plt.scatter(X_embedded[idx_real, 0], X_embedded[idx_real, 1], 
                    marker='o', s=80, edgecolors='black', linewidth=1.0, alpha=0.9, 
                    color=colors[digit], label=f'Digit {digit}')

    plt.legend(loc='best', title="Real Data Classes")
    plt.title(f"t-SNE with Class Labels (Beta={CONFIG['BETA']})", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"real_vs_fake_labeled_beta{CONFIG['BETA']}.png", dpi=300)
    print(f"[Done] Labeled t-SNE saved.")
    
    # Outlier Analysis
    print("\n[Analysis] Finding top outliers...")
    fig, axes = plt.subplots(10, 6, figsize=(12, 20))
    plt.subplots_adjust(hspace=0.5)
    
    for digit in range(10):
        idx_real_digit = (y_all == digit) & (m_all == 'Real')
        if np.sum(idx_real_digit) == 0: continue
        
        centroid = np.mean(X_embedded[idx_real_digit], axis=0)
        
        def get_dist(idx): return np.linalg.norm(X_embedded[idx] - centroid)

        real_indices = np.where(idx_real_digit)[0]
        sorted_real = sorted(real_indices, key=get_dist, reverse=True)
        worst_real_idx = sorted_real[0]
        
        idx_fake_digit = (y_all == digit) & (m_all == 'Fake')
        fake_indices = np.where(idx_fake_digit)[0]
        sorted_fake = sorted(fake_indices, key=get_dist, reverse=True)
        worst_fake_indices = sorted_fake[:5]
        
        ax = axes[digit, 0]
        img = images_list[worst_real_idx].squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Real {digit}\n(Outlier)", fontsize=10, color='blue')
        ax.axis('off')
        
        for i, fake_idx in enumerate(worst_fake_indices):
            ax = axes[digit, i+1]
            img = images_list[fake_idx].squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Fake {digit}\n(Rank {i+1})", fontsize=10, color='red')
            ax.axis('off')

    plt.suptitle(f"Top Outliers Analysis (Beta={CONFIG['BETA']})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"outliers_analysis_beta{CONFIG['BETA']}.png", dpi=300)
    print(f"[Done] Outlier images saved.")
    plt.show()
    
def analyze_feature_uncertainty(model, save_path="feature_uncertainty.png"):
    """
    [기능 6] Digit별 Feature 예측의 불확실성(Sigma) 정량화
    """
    print(f"\n[Uncertainty Analysis] Calculating model confidence (Sigma)...")
    model.eval()
    
    results = []
    
    with torch.no_grad():
        # 각 숫자(0~9)에 대해 예측된 분포 확인
        t_all = torch.eye(10).to(CONFIG["DEVICE"]) # (10, 10) - One-hot for 0..9
        
        h = model.morph_predictor_shared(t_all)
        mu = model.morph_predictor_mu(h)
        logvar = model.morph_predictor_logvar(h)
        sigma = torch.exp(0.5 * logvar) # Standard Deviation (Uncertainty)
        
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        
        # DataFrame 생성
        print("\n=== Feature Prediction Uncertainty (Sigma) ===")
        print("값이 작을수록 모델이 해당 피처값에 대해 확신하고 있음을 의미\n")
        
        for digit in range(10):
            for i, feat_name in enumerate(FEATURE_NAMES):
                # uncertainty = sigma
                results.append({
                    "Digit": digit,
                    "Feature": feat_name,
                    "Mean": mu_np[digit, i],
                    "Uncertainty (Sigma)": sigma_np[digit, i]
                })
                
    df = pd.DataFrame(results)
    
    # CSV 저장
    df.to_csv("feature_uncertainty.csv", index=False)
    print(" -> Saved numerical results to 'feature_uncertainty.csv'")

    # 시각화: Heatmap
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot(index="Feature", columns="Digit", values="Uncertainty (Sigma)")
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'Sigma (Uncertainty)'})
    plt.title("Model Uncertainty by Feature and Digit (Lower is Better Confidence)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" -> Saved visualization to '{save_path}'")
    
    # 텍스트 요약 출력
    # 각 숫자별로 가장 불확실한 피처 1개씩 출력
    print("\n[Summary] Most Uncertain Features per Digit:")
    for digit in range(10):
        digit_df = df[df["Digit"] == digit]
        max_row = digit_df.loc[digit_df["Uncertainty (Sigma)"].idxmax()]
        min_row = digit_df.loc[digit_df["Uncertainty (Sigma)"].idxmin()]
        print(f"  Digit {digit}: Most Uncertain = {max_row['Feature']} ({max_row['Uncertainty (Sigma)']:.3f}) | Most Confident = {min_row['Feature']} ({min_row['Uncertainty (Sigma)']:.3f})")
