import os
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import load_image_any, normalize_image

class ImageTableDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root, 
        image_ext: str = "tiff",
        resize_hw: tuple = (384, 640),
        clip_percentile: float = 99.5,
    ):
        self.df = df.reset_index(drop=True)
        self.resize_hw = resize_hw
        self.clip_percentile = clip_percentile

        if isinstance(image_root, str):
            self.image_roots = [image_root]
        else:
            self.image_roots = image_root

        self.path_map = {}
        print(f"Scanning directories: {self.image_roots}")
        
        for r in self.image_roots:
            # 확장자 상관없이 모든 파일 검색 후 필터링
            all_files = glob.glob(os.path.join(r, "**", "*"), recursive=True)
            for f in all_files:
                # tiff, tif, TIFF, TIF 모두 허용
                if f.lower().endswith(('.tiff', '.tif')):
                    try:
                        # 파일명 파싱: "blah-504002.vessel.tiff" -> "504002"
                        fname = os.path.basename(f)
                        # .tiff, .vessel 등 뒤에서부터 제거
                        clean_name = fname.replace(".tiff", "").replace(".tif", "").replace(".vessel", "")
                        img_id = clean_name.split("-")[-1]
                        
                        self.path_map[img_id] = f
                    except:
                        continue
                    
        print(f"Found {len(self.path_map)} images.")

        self.df['Image ID'] = self.df['Image ID'].astype(str)
        initial_len = len(self.df)
        self.df = self.df[self.df['Image ID'].isin(self.path_map.keys())].reset_index(drop=True)
        print(f"Matched {len(self.df)} / {initial_len} rows.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = str(row['Image ID'])
        path = self.path_map.get(img_id, None)
        
        if path is None:
            img = torch.zeros((1, self.resize_hw[0], self.resize_hw[1]))
        else:
            arr = load_image_any(path)
            img = normalize_image(arr, self.clip_percentile)
        
        if self.resize_hw:
            img = F.interpolate(img.unsqueeze(0), size=self.resize_hw, mode='bilinear', align_corners=False)
            img = img.squeeze(0)

        return {"x": img, "id": img_id}