import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
import tifffile
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from config import CONFIG
from tqdm import tqdm

class VesselDataset(Dataset):
    """
    Vessel Analysis Dataset
    - Loads CSV (Features & Conditions)
    - Loads Images on demand (to save RAM)
    - Resizes to (H, W) defined in CONFIG
    """
    def __init__(self, train=True, transform=None):
        self.csv_path = CONFIG["DATA_CSV"]
        self.data_root = CONFIG["DATA_ROOT"]
        self.train = train
        
        # 1. Load CSV
        print(f"[Dataset] Loading CSV: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        # 2. Filter Valid Images
        # We need to find the actual file path for each Image ID
        # Image ID: 504002 -> needs filename mapping
        # Scanning directory...
        self.valid_data = []
        
        print(f"[Dataset] Scanning image files in {self.data_root}...")
        # Finding all .vessel.tiff files recursively
        # Pattern: data/Plate-*/Row/Cell-ImageID.vessel.tiff
        # Optimization: Create a dict mapping ImageID -> Path
        
        # Assuming the structure is consistent:
        # We can glob all *vessel.tiff files and extract ID
        all_files = glob.glob(os.path.join(self.data_root, "**", "*.vessel.mip.tiff"), recursive=True)
        img_id_to_path = {}
        
        for fpath in all_files:
            # Filename example: "H11-503938.vessel.tiff"
            basename = os.path.basename(fpath)
            # Extract 503938
            try:
                # Split by '-', take last part, split by '.', take first
                # H11-503938.vessel.tiff -> 503938
                parts = basename.split('-')[-1]
                img_id = int(parts.split('.')[0])
                img_id_to_path[img_id] = fpath
            except:
                continue
                
        # Match CSV with Files
        found_count = 0
        missing_count = 0
        
        self.feature_cols = [
            "Node count", "Extremity Count", "Junction Count", "Edge count", 
            "Segment Count", "Branch Count", "Isolated Edge Count", 
            "Subnetwork Count(edge count >= 3)", "Total Vessel Length (μm)", 
            "Mean Tortuosity", "Total Vessel Volume (μm^3)", "Average Vessel Radius (μm)"
        ]
        
        # Prepare T (Group Name) Mapping
        # Unique Groups
        self.group_names = sorted(self.df['group_name'].dropna().unique())
        self.group_to_idx = {name: i for i, name in enumerate(self.group_names)}
        print(f"[Dataset] Found {len(self.group_names)} unique groups (T).")
        
        # Collect valid samples
        temp_data = []
        for _, row in self.df.iterrows():
            img_id = row['Image ID']
            if img_id in img_id_to_path:
                fpath = img_id_to_path[img_id]
                
                # Check NaNs in features
                m_values = row[self.feature_cols].values.astype(float)
                if np.isnan(m_values).any():
                    continue
                    
                group = row['group_name']
                if pd.isna(group):
                    continue
                    
                t_idx = self.group_to_idx[group]
                
                temp_data.append({
                    "path": fpath,
                    "m": m_values,
                    "t": t_idx
                })
                found_count += 1
            else:
                missing_count += 1
                
        print(f"[Dataset] Matched: {found_count}, Missing/Invalid: {missing_count}")
        
        # 3. Normalize M Features
        # Using StandardScaler fitted on ALL data (simplification for now)
        # Ideally fit on Train only, but since this is verification.
        all_m = np.array([item['m'] for item in temp_data])
        self.scaler = StandardScaler()
        self.scaler.fit(all_m)
        self.norm_m = self.scaler.transform(all_m)
        
        if np.isnan(self.norm_m).any():
            print(f"[Warning] NaNs found in normalized M features (fitting stage).")

        for i, item in enumerate(temp_data):
            item['m_norm'] = self.norm_m[i]
            
        # 4. Train/Val Split (80/20) based on simple indexing for now
        # Or random split
        np.random.seed(42)
        indices = np.random.permutation(len(temp_data))
        split_point = int(len(temp_data) * 0.8)
        
        if self.train:
            self.indices = indices[:split_point]
            print(f"[Dataset] Training Set: {len(self.indices)} samples")
        else:
            self.indices = indices[split_point:]
            print(f"[Dataset] Validation Set: {len(self.indices)} samples")
            
        self.data_source = [temp_data[i] for i in self.indices]

        # 5. Transforms
        self.img_h = CONFIG["IMG_HEIGHT"]
        self.img_w = CONFIG["IMG_WIDTH"]
        
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_h, self.img_w), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5), # Left-Right Mirror
                transforms.RandomVerticalFlip(p=0.5),   # Up-Down Mirror
                transforms.RandomRotation(degrees=180), # Rotate +/- 180 (Full rotation)
                # transforms.ToTensor() moved to manual
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_h, self.img_w), antialias=True),
                # transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        # Load Image using tifffile (Handle float32 correctly)
        try:
            img = tifffile.imread(item['path']) # (H, W) float32
        except:
            # Fallback to PIL if tifffile misses
            img = np.array(Image.open(item['path']))
        
        # Normalize to [0, 1] per image
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img)
            
        # Convert to Tensor (1, H, W)
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)
        
        # Binarize (Simple Adaptive Threshold)
        # Reference used fixed threshold, but for normalized data, dynamic is safer
        # Keep it continuous 0.0 - 1.0 or Hard Binary 0.0 / 1.0?
        # User said "Binarize", assuming Hard Binary.
        threshold = img_tensor.mean() 
        img_tensor = (img_tensor > threshold).float()
            
        # Apply Geometric Transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Prepare M
        m_tensor = torch.tensor(item['m_norm'], dtype=torch.float32)
        
        # Prepare T (One-hot)
        t_idx = item['t']
        t_onehot = torch.zeros(CONFIG["T_DIM"], dtype=torch.float32)
        t_onehot[t_idx] = 1.0
        
        # Return
        return img_tensor, m_tensor, t_onehot
