import os
import glob
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tifffile
from skimage import transform
import albumentations as A
import cv2

class CausalDataset(Dataset):
    def __init__(self, csv_path, img_root_dirs, img_size=(512, 960), is_train=True):
        """
        img_root_dirs: 폴더 경로 문자열 1개 또는 리스트([]) 모두 지원
        """
        # 1. 경로가 문자열 하나로 오면 리스트로 감싸기 
        if isinstance(img_root_dirs, str):
            img_root_dirs = [img_root_dirs]
            
        self.img_root_dirs = img_root_dirs
        self.img_size = img_size
        self.is_train = is_train
        
        # 2. CSV 로드
        self.data = pd.read_csv(csv_path)
        
        # 3. 이미지 파일 리스트 확보 (여러 폴더 반복 검색)
        self.image_paths = []
        print(f"Scanning directories: {self.img_root_dirs}")
        
        for root_dir in self.img_root_dirs:
            # 혹시 확장자가 .vessel.tiff 일 수도 있으므로 *.tiff로 검색하면 다 잡히게
            files = glob.glob(os.path.join(root_dir, "**", "*.vessel.tiff"), recursive=True)
            self.image_paths.extend(files)
            
        print(f"Found total {len(self.image_paths)} tiff files.")

        # 4. CSV - 이미지 매칭 (ID 기준)
        self.path_map = {}
        for p in self.image_paths:
            try:
                # 파일명: plate-25250-01-504002.vessel.tiff -> 504002 추출
                fname = os.path.basename(p)
                # .tiff 제거 -> .vessel 제거 (있으면) -> split
                name_no_ext = fname.replace(".tiff", "").replace(".vessel", "")
                img_id_str = name_no_ext.split("-")[-1]
                
                self.path_map[str(img_id_str)] = p
            except:
                continue

        # 5. 매칭되는 데이터만 남기기
        self.data['Image ID'] = self.data['Image ID'].astype(str)
        initial_len = len(self.data)
        self.data = self.data[self.data['Image ID'].isin(self.path_map.keys())].reset_index(drop=True)
        print(f"Matched {len(self.data)} images out of {initial_len} CSV rows.")

        if len(self.data) == 0:
            print("매칭된 이미지 0개")

        # 6. T (Group) 매핑
        self.t_col = 'group_name'
        self.groups = sorted(self.data[self.t_col].unique())
        self.group_to_idx = {g: i for i, g in enumerate(self.groups)}
        self.num_groups = len(self.groups)

        # 7. M (Morphological Features) 컬럼
        self.m_cols = [
            'Node count', 'Extremity Count', 'Junction Count', 'Edge count', 
            'Segment Count', 'Branch Count', 'Isolated Edge Count', 
            'Subnetwork Count(edge count >= 3)', 'Total Vessel Length (μm)', 
            'Mean Tortuosity', 'Total Vessel Volume (μm^3)', 'Average Vessel Radius (μm)'
        ]
        
        # M 정규화
        m_values = self.data[self.m_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.m_min = m_values.min(axis=0)
        self.m_max = m_values.max(axis=0)
        diff = self.m_max - self.m_min
        diff[diff == 0] = 1.0
        self.m_denom = diff

        # 8. Augmentation
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
        """한 장씩 읽어서 Max Projection"""
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
            # 이미지 로드 (Safe Mode)
            image = self.load_mip_safe(img_path)
            
            # 전처리
            image = np.clip(image, image.min(), 3000)
            if image.shape[0] > 200:
                image = image[100:-100, :]

            image = transform.resize(image, self.img_size, anti_aliasing=True)
            image = image.astype("float32") # Float32 변환

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-5)
            
            # Tensor 변환 (Float 강제)
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float().unsqueeze(0) 
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
    