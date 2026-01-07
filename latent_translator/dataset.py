from typing import Optional, Tuple, Dict
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import infer_image_path, load_image_any, normalize_image

class ImageTableDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        image_ext: str,
        image_path_col: Optional[str] = None,
        resize_hw: Optional[Tuple[int, int]] = None,
        clip_percentile: float = 99.5,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.image_ext = image_ext
        self.image_path_col = image_path_col
        self.resize_hw = resize_hw
        self.clip_percentile = clip_percentile

    def __len__(self) -> int:
        return len(self.df)

    def _get_path(self, row: pd.Series) -> str:
        if self.image_path_col and self.image_path_col in row.index:
            return str(row[self.image_path_col])
        return infer_image_path(row, self.image_root, self.image_ext)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = self._get_path(row)
        img = load_image_any(path)
        img = normalize_image(img, clip_percentile=self.clip_percentile)

        if self.resize_hw is not None:
            t = torch.from_numpy(img)[None, ...]
            t = F.interpolate(t, size=self.resize_hw, mode="bilinear", align_corners=False)
            img = t[0].numpy()

        x = torch.from_numpy(img)
        return {"x": x}
    