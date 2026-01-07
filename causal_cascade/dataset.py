import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tifffile
from skimage import transform, util
import albumentations as A
import cv2

class CausalDataset(Dataset):
    def __init__(self, csv_path, img_root_dir, img_size=(384, 640), is_train=True):
        self.img_root_dir = img_root_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # 1. CSV 로드
        self.data = pd.read_csv(csv_path)
        
        # 2. 이미지 파일 리스트 확보
        self.image_paths = sorted(
            glob.glob(os.path.join(img_root_dir, "**", "*.tiff"), recursive=True)
        )

        # 3. 매칭
        self.path_map = {}
        for p in self.image_paths:
            try:
                img_id_str = os.path.basename(p).split(".")[0].split("-")[-1]
                self.path_map[str(img_id_str)] = p
            except:
                continue

        self.data['Image ID'] = self.data['Image ID'].astype(str)
        self.data = self.data[self.data['Image ID'].isin(self.path_map.keys())].reset_index(drop=True)

        # 4. T (Group)
        self.t_col = 'group_name'
        self.groups = sorted(self.data[self.t_col].unique())
        self.group_to_idx = {g: i for i, g in enumerate(self.groups)}
        self.num_groups = len(self.groups)

        # 5. M (Features)
        self.m_cols = [
            'Node count', 'Extremity Count', 'Junction Count', 'Edge count', 
            'Segment Count', 'Branch Count', 'Isolated Edge Count', 
            'Subnetwork Count(edge count >= 3)', 'Total Vessel Length (μm)', 
            'Mean Tortuosity', 'Total Vessel Volume (μm^3)', 'Average Vessel Radius (μm)'
        ]
        
        m_values = self.data[self.m_cols].values
        self.m_min = m_values.min(axis=0)
        self.m_max = m_values.max(axis=0)
        diff = self.m_max - self.m_min
        diff[diff == 0] = 1.0
        self.m_denom = diff

        # 6. Augmentation
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.01, 0.1), contrast_limit=(-0.01, 0.05), p=0.5
                ),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def load_mip_safe(self, path):
        with tifffile.TiffFile(path) as tif:
            projection = tif.pages[0].asarray()
            for page in tif.pages[1:]:
                np.maximum(projection, page.asarray(), out=projection)
        return projection

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(row['Image ID'])
        img_path = self.path_map[img_id]

        try:
            image = self.load_mip_safe(img_path)
            
            image = np.clip(image, image.min(), 3000)
            if image.shape[0] > 200:
                image = image[100:-100, :]

            image = transform.resize(image, self.img_size, anti_aliasing=True)
            image = image.astype("float32") # 여기서 float32로 변환

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-5)
            
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float().unsqueeze(0) # .float() 추가
            else:
                image = image.float()

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros((1, self.img_size[0], self.img_size[1]), dtype=torch.float32)

        # --- M (Features) ---
        m_raw = row[self.m_cols].values.astype(np.float32)
        m_norm = (m_raw - self.m_min) / self.m_denom
        m_tensor = torch.tensor(m_norm).float() 

        # --- T (Condition) ---
        t_label = self.group_to_idx[row[self.t_col]]
        t_tensor = torch.tensor(t_label, dtype=torch.long)

        return image, m_tensor, t_tensor
    