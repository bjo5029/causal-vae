from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

# LOOCV 방식 사용
def fit_translator_ridge(
    Z: np.ndarray,
    M: np.ndarray,
    # Z_test, M_test는 LOOCV에서는 쓰지 않지만 함수 형태 유지를 위해 받음 (무시됨)
    Z_test: np.ndarray = None, 
    M_test: np.ndarray = None,
    feature_names: List[str] = [],
    alpha: float = 1.0,
) -> Tuple[Ridge, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    LOOCV를 적용하여 Ridge 번역기를 학습하고 평가
    데이터가 극소량(N=16)일 때 적합함
    """
    # 1. LOOCV 수행 (성능 평가용)
    loo = LeaveOneOut()
    y_trues = []
    y_preds = []
    
    # 전체 데이터(Z, M)를 합쳐서 사용 (train/test 인자 무시)
    # 함수 호출 시 Z_train 대신 전체 Z, M_train 대신 전체 M을 넣어야 함

    # 여기서는 들어온 Z, M이 "전체 데이터"라고 가정
    print(f"Running LOOCV on {len(Z)} samples...")
    
    # 16번 반복
    for train_ix, test_ix in loo.split(Z):
        X_tr, X_te = Z[train_ix], Z[test_ix]
        y_tr, y_te = M[train_ix], M[test_ix]
        
        # 모델 학습
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
        model.fit(X_tr, y_tr)
        
        # 예측
        pred = model.predict(X_te)
        
        y_trues.append(y_te[0]) # (n_features,)
        y_preds.append(pred[0]) # (n_features,)

    y_trues = np.array(y_trues) # (N, n_features)
    y_preds = np.array(y_preds) # (N, n_features)

    # 2. 성능 지표 계산
    metrics = []
    for j, fn in enumerate(feature_names):
        y_t = y_trues[:, j]
        y_p = y_preds[:, j]
        
        # R2 Score
        r2 = r2_score(y_t, y_p)
        
        # Correlation
        if np.std(y_t) < 1e-12 or np.std(y_p) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(y_t, y_p)[0, 1])
            
        metrics.append({"feature": fn, "r2": float(r2), "corr": corr})

    metrics_df = pd.DataFrame(metrics).sort_values("r2", ascending=False).reset_index(drop=True)

    # 3. 최종 번역기 (전체 데이터로 학습)
    # 분석에 쓸 W(가중치)는 데이터 16개를 모두 보고 배운 게 제일 정확함
    final_model = Ridge(alpha=alpha, fit_intercept=True, random_state=0)
    final_model.fit(Z, M)
    
    W = final_model.coef_
    
    # Mhat_test 자리에 그냥 전체 예측값을 반환 (코드 호환성 위해)
    Mhat_full = final_model.predict(Z)

    return final_model, metrics_df, Mhat_full, W

def group_split_indices(df: pd.DataFrame, group_col: str, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col].astype(str).values
    idx = np.arange(len(df))
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    return train_idx, test_idx

def compute_group_means(arr: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    uniq = pd.unique(groups)
    rows = []
    for g in uniq:
        m = arr[groups == g].mean(axis=0)
        rows.append([g] + list(m))
    cols = ["group"] + [f"dim_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(rows, columns=cols)

def pairwise_contrasts(groups: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pairs.append((groups[i], groups[j]))
    return pairs

def contrast_delta(group_means: pd.DataFrame, g1: str, g2: str) -> np.ndarray:
    row1 = group_means[group_means["group"] == g1].iloc[0]
    row2 = group_means[group_means["group"] == g2].iloc[0]
    dims = [c for c in group_means.columns if c.startswith("dim_")]
    return row2[dims].to_numpy(dtype=np.float32) - row1[dims].to_numpy(dtype=np.float32)

def topk_features(delta_m: np.ndarray, feature_names: List[str], k: int = 10) -> List[Tuple[str, float]]:
    absd = np.abs(delta_m)
    idx = np.argsort(-absd)[:k]
    return [(feature_names[i], float(delta_m[i])) for i in idx]

def bootstrap_feature_stability(
    df: pd.DataFrame,
    Z: np.ndarray,
    M: np.ndarray,
    feature_names: List[str],
    group_col: str,
    translator: Ridge,
    scaler_m: StandardScaler,
    n_boot: int = 200,
    top_k: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    groups = df[group_col].astype(str).values
    uniq_groups = sorted(pd.unique(groups).tolist())
    pairs = pairwise_contrasts(uniq_groups)

    counts = {fn: 0 for fn in feature_names}
    total_slots = 0

    for b in range(n_boot):
        idx = rng.randint(0, len(df), size=len(df))
        Zb = Z[idx]
        gb = groups[idx]
        Z_means = compute_group_means(Zb, gb)

        for (g1, g2) in pairs:
            # Predict M at mean Z
            z1 = Z_means[Z_means["group"] == g1].iloc[0][[c for c in Z_means.columns if c.startswith("dim_")]].to_numpy()
            z2 = Z_means[Z_means["group"] == g2].iloc[0][[c for c in Z_means.columns if c.startswith("dim_")]].to_numpy()
            m1_hat_scaled = translator.predict(z1.reshape(1, -1))[0]
            m2_hat_scaled = translator.predict(z2.reshape(1, -1))[0]
            dm_hat_scaled = m2_hat_scaled - m1_hat_scaled
            
            # Convert to original units
            dm_hat = dm_hat_scaled * scaler_m.scale_
            top = topk_features(dm_hat, feature_names, k=top_k)
            for fn, _val in top:
                counts[fn] += 1
            total_slots += top_k

    out = pd.DataFrame({
        "feature": list(counts.keys()),
        "topk_count": list(counts.values()),
    })
    out["stability_freq"] = out["topk_count"] / max(total_slots, 1)
    out = out.sort_values("stability_freq", ascending=False).reset_index(drop=True)
    return out
