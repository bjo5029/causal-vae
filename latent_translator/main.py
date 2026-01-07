import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from utils import set_seed, safe_mkdir
from dataset import ImageTableDataset
import analysis

from models import ViTVAE 
from engine import train_vit_vae, extract_vit_latents

def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--table_csv", type=str, required=True, help="메타데이터 CSV 경로")
    ap.add_argument("--out_dir", type=str, required=True, help="결과 저장 폴더")
    ap.add_argument("--image_root", type=str, required=True, help="이미지 폴더 최상위 경로")
    ap.add_argument("--checkpoint", type=str, default="", help="학습된 ViTVAE 체크포인트(.pth) 경로")

    # 데이터 관련
    ap.add_argument("--image_ext", type=str, default="npy")
    ap.add_argument("--image_path_col", type=str, default="")
    ap.add_argument("--group_col", type=str, default="group_name")
    ap.add_argument("--split_group_col", type=str, default="Plate")
    ap.add_argument("--drop_cols", type=str, default="Plate,Well,Image ID,group_name,Chip Barcode,Well Number")

    # ViTVAE 모델 설정
    ap.add_argument("--resize_h", type=int, default=768, help="ViTVAE 고정 높이")
    ap.add_argument("--resize_w", type=int, default=1280, help="ViTVAE 고정 너비")
    ap.add_argument("--z_dim", type=int, default=128, help="Latent dimension (모델과 일치해야 함)")
    
    # 학습 설정 (체크포인트 로드 시 0으로 설정하여 학습 건너뛰기)
    ap.add_argument("--vae_epochs", type=int, default=0, help="추가 학습할 에폭 수 (0이면 학습 생략)")
    ap.add_argument("--vae_batch_size", type=int, default=4, help="메모리 문제로 작게 설정 권장")
    ap.add_argument("--vae_lr", type=float, default=1e-4)
    ap.add_argument("--vae_beta", type=float, default=1.0)
    
    # 분석 설정
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=10)

    args = ap.parse_args()

    set_seed(args.seed)
    safe_mkdir(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # 1. 데이터 로드 및 전처리
    # ---------------------------------------------------------
    print(f"Loading data from {args.table_csv}...")
    df = pd.read_csv(args.table_csv)
    image_path_col = None if args.image_path_col.strip() == "" else args.image_path_col.strip()

    # Feature 컬럼 발굴
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip() != ""]
    feature_cols = []
    for c in df.columns:
        if c in drop_cols: continue
        if c in ["Plate", "Well", "Image ID", args.group_col, args.split_group_col]: continue
        if image_path_col is not None and c == image_path_col: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError("분석할 숫자형 Feature 컬럼이 없음")

    # 결측치 처리 및 스케일링 (M)
    df_feat = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.fillna(df_feat.median(numeric_only=True))
    M_raw = df_feat.to_numpy(dtype=np.float32)

    scaler_m = StandardScaler()
    M_scaled = scaler_m.fit_transform(M_raw)

    # ---------------------------------------------------------
    # 2. Dataset & DataLoader
    # ---------------------------------------------------------
    resize_hw = (args.resize_h, args.resize_w)
    print(f"Image resize target: {resize_hw}")
    
    ds = ImageTableDataset(
        df=df,
        image_root=args.image_root,
        image_ext=args.image_ext,
        image_path_col=image_path_col,
        resize_hw=resize_hw,
    )
    
    # 채널 수 확인 (첫 이미지 로드)
    first_x = ds[0]["x"]
    in_ch = first_x.shape[0]
    print(f"Detected input channels: {in_ch}")

    loader = DataLoader(ds, batch_size=args.vae_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_noshuf = DataLoader(ds, batch_size=args.vae_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---------------------------------------------------------
    # 3. ViTVAE 모델 초기화 및 체크포인트 로드
    # ---------------------------------------------------------
    print("Initializing ViTVAE...")
    model = ViTVAE(
        in_channels=in_ch,
        latent_dim=args.z_dim,
        img_size=resize_hw,  # (768, 1280)
        patch_size=32
    ).to(device)

    # 체크포인트 로드
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # state_dict가 dict 안에 래핑되어 있는지 확인
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully.")
        else:
            print(f"[Warning] Checkpoint path provided but file not found: {args.checkpoint}")
            print("Initializing with random weights.")
    else:
        print("No checkpoint provided. Initializing with random weights.")

    # ---------------------------------------------------------
    # 4. 학습 (옵션) 및 Latent 추출
    # ---------------------------------------------------------
    if args.vae_epochs > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.vae_lr)
        train_vit_vae(model, loader, optimizer, device, args.vae_epochs, beta=args.vae_beta)
    else:
        print("Skipping training (vae_epochs=0). Using loaded/random weights.")

    # Z 추출
    Z = extract_vit_latents(model, loader_noshuf, device)
    print(f"Latents extracted. Shape: {Z.shape}")

    # ---------------------------------------------------------
    # 5. 분석 파이프라인 (Ridge Translator & Contrasts)
    # ---------------------------------------------------------
    # Train/Test Split
    train_idx, test_idx = analysis.group_split_indices(
        df, group_col=args.split_group_col, test_size=args.test_size, seed=args.seed
    )
    Z_train, Z_test = Z[train_idx], Z[test_idx]
    M_train, M_test = M_scaled[train_idx], M_scaled[test_idx]

    # Translator 학습
    print("Fitting Translator (Z -> M)...")
    # Z와 M_scaled 전체를 넣음
    translator, metrics_df, _, W = analysis.fit_translator_ridge(
    Z, M_scaled, None, None, feature_cols, args.ridge_alpha
)
    
    metrics_path = os.path.join(args.out_dir, "translator_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Top 5 explainable features:\n{metrics_df.head(5)}")

    # Sample Level 결과 저장
    out_sample = df[["Plate", "Well", "Image ID", args.group_col, args.split_group_col]].copy()
    z_cols = [f"z_{i:02d}" for i in range(Z.shape[1])]
    
    # 데이터프레임 생성 효율화
    z_df = pd.DataFrame(Z, columns=z_cols)
    out_sample = pd.concat([out_sample, z_df], axis=1)
    
    # 원본 M값 붙이기
    for j, c in enumerate(feature_cols):
        out_sample[c] = M_raw[:, j]
        
    out_sample.to_csv(os.path.join(args.out_dir, "sample_level.csv"), index=False)

    # ---------------------------------------------------------
    # 6. Contrast Analysis & Ranking
    # ---------------------------------------------------------
    print("Running contrast analysis...")
    groups = df[args.group_col].astype(str).values
    Z_means = analysis.compute_group_means(Z, groups)
    M_means = analysis.compute_group_means(M_raw, groups)
    uniq_groups = sorted(pd.unique(groups).tolist())
    pairs = analysis.pairwise_contrasts(uniq_groups)

    # Helper: Z -> M_original 예측
    def pred_M_original(z_vec):
        m_scaled = translator.predict(z_vec.reshape(1, -1))[0]
        return (m_scaled * scaler_m.scale_) + scaler_m.mean_

    contrast_rows = []
    agg_abs = np.zeros(len(feature_cols), dtype=np.float64)

    for (g1, g2) in pairs:
        # Latent Space Difference
        dz = analysis.contrast_delta(Z_means, g1, g2)
        
        # Translate Mean Z difference to M difference
        z1 = Z_means[Z_means["group"] == g1].iloc[0, 1:].to_numpy(dtype=np.float32)
        z2 = Z_means[Z_means["group"] == g2].iloc[0, 1:].to_numpy(dtype=np.float32)
        
        dm_hat = pred_M_original(z2) - pred_M_original(z1)
        agg_abs += np.abs(dm_hat)

        # Observed Difference (Ground Truth)
        m1_obs = M_means[M_means["group"] == g1].iloc[0, 1:].to_numpy(dtype=np.float32)
        m2_obs = M_means[M_means["group"] == g2].iloc[0, 1:].to_numpy(dtype=np.float32)
        dm_obs = m2_obs - m1_obs

        # Top-k Features
        top = analysis.topk_features(dm_hat, feature_cols, k=args.top_k)
        top_idx = [feature_cols.index(fn) for fn, _ in top]
        
        # Direction Agreement (Sign match)
        sign_agree = np.mean(np.sign(dm_hat[top_idx]) == np.sign(dm_obs[top_idx]))

        contrast_rows.append({
            "contrast": f"{g2} - {g1}",
            "group_1": g1,
            "group_2": g2,
            "dz_l2": float(np.linalg.norm(dz)),
            "top_features_json": json.dumps(top, ensure_ascii=False),
            "sign_agreement": float(sign_agree)
        })

    # 저장
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(os.path.join(args.out_dir, "contrast_results.csv"), index=False)

    # Feature Ranking
    n_con = max(len(pairs), 1)
    rank_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_importance": (agg_abs / n_con).astype(np.float32),
    }).sort_values("mean_abs_importance", ascending=False).reset_index(drop=True)

    # Add Explainability (R2) Info
    rank_df = rank_df.merge(metrics_df[["feature", "r2"]], on="feature", how="left")
    
    # Bootstrap Stability (Optional - 시간 좀 걸림)
    print("Running bootstrap stability analysis...")
    stab_df = analysis.bootstrap_feature_stability(
        df, Z, M_raw, feature_cols, args.group_col, translator, scaler_m, 
        n_boot=args.n_boot, top_k=args.top_k, seed=args.seed
    )
    rank_df = rank_df.merge(stab_df[["feature", "stability_freq"]], on="feature", how="left").fillna(0.0)

    rank_df.to_csv(os.path.join(args.out_dir, "feature_ranking.csv"), index=False)

    # Summary
    summary = {
        "n_samples": int(len(df)),
        "z_dim": int(args.z_dim),
        "avg_r2": float(np.nanmean(metrics_df["r2"].values)),
        "checkpoint": args.checkpoint
    }
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"All Done. Results in {args.out_dir}")

if __name__ == "__main__":
    main()
