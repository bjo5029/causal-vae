import os
import numpy as np
import torch
import pandas as pd
import tifffile
import random

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_image_any(path: str) -> np.ndarray:
    """
    3D Tiff 이미지를 읽어서 2D MIP로 변환하고 Float32로 반환
    """
    try:
        # 1. 파일 로딩
        if path.lower().endswith(('.tif', '.tiff')):
            arr = tifffile.imread(path)
        elif path.lower().endswith('.npy'):
            arr = np.load(path)
        else:
            import imageio.v2 as imageio
            arr = imageio.imread(path)
            
        # 2. 3D -> 2D MIP (Max Intensity Projection)
        if arr.ndim == 3:
            arr = np.max(arr, axis=0)
            
        return arr.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # 에러 방지용 더미 이미지 (검은색)
        return np.zeros((100, 100), dtype=np.float32)

def normalize_image(arr: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Robust Min-Max Normalization -> Tensor 변환
    """
    if arr.size == 0: return torch.zeros(1, 10, 10)

    # 1. Clip outliers
    vmin, vmax = np.percentile(arr, [100 - clip_percentile, clip_percentile])
    arr = np.clip(arr, vmin, vmax)
    
    # 2. Scale [0, 1]
    denom = vmax - vmin
    if denom == 0: denom = 1e-5
    arr = (arr - vmin) / denom
    
    # 3. To Tensor (C, H, W)
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    return tensor
