# utils.py
import os
import numpy as np
import torch
import pandas as pd

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def infer_image_path(row: pd.Series, image_root: str, ext: str) -> str:
    plate = str(row["Plate"])
    well = str(row["Well"])
    img_id = str(row["Image ID"])
    fname = f"{well}-{img_id}.{ext}"
    return os.path.join(image_root, plate, well, fname)

def load_image_any(path: str) -> np.ndarray:
    """
    이미지를 float32 numpy array로 로드.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)

    try:
        if ext in [".tif", ".tiff"]:
            import tifffile
            arr = tifffile.imread(path)
            return arr.astype(np.float32)
        else:
            import imageio.v2 as imageio
            arr = imageio.imread(path)
            return arr.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {path} ({e})")

def normalize_image(arr: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    x = arr
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):  # (H,W,C)
        x = np.transpose(x, (2, 0, 1))  # (C,H,W)
    elif x.ndim == 2:
        x = x[None, ...]  # (1,H,W)

    x = x.astype(np.float32)
    hi = np.percentile(x, clip_percentile)
    if hi <= 0:
        hi = float(np.max(x)) if np.max(x) > 0 else 1.0
    x = np.clip(x, 0, hi) / hi
    return x
