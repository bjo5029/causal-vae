import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from skimage.measure import label as sk_label, regionprops 
from scipy.ndimage import distance_transform_edt

def extract_12_features(img_tensor):
    """이미지에서 12가지 형태학적(Morphological) 특징 추출"""
    img = img_tensor.squeeze().numpy()
    binary = img > 0.2
    
    if np.sum(binary) == 0: return torch.zeros(12, dtype=torch.float32)

    labeled_img = sk_label(binary)
    props = regionprops(labeled_img)
    
    if not props: return torch.zeros(12, dtype=torch.float32)
    
    # 가장 큰 덩어리만 선택 (노이즈 제거)
    prop = props[np.argmax([p.area for p in props])]

    # 1. 기하학적 특징 계산
    f1_area = prop.area / 784.0
    f2_perimeter = prop.perimeter / 100.0
    dist_map = distance_transform_edt(binary)
    f3_thickness = np.max(dist_map) / 5.0
    f4_major_axis = prop.major_axis_length / 28.0
    f5_eccentricity = prop.eccentricity
    f6_orientation = (prop.orientation + np.pi/2) / np.pi
    f7_solidity = prop.solidity
    f8_extent = prop.extent
    
    minr, minc, maxr, maxc = prop.bbox
    height = maxr - minr
    width = maxc - minc
    f9_aspect_ratio = (width / height) if height > 0 else 0
    f9_aspect_ratio = np.clip(f9_aspect_ratio, 0, 3.0) / 3.0 
    
    # 2. 위상학적 특징 (Topology)
    f10_euler = (prop.euler_number + 2) / 4.0 
    
    # 3. 대칭성 특징
    flipped_lr = np.fliplr(img)
    f11_h_symmetry = 1.0 - np.mean(np.abs(img - flipped_lr))
    flipped_ud = np.flipud(img)
    f12_v_symmetry = 1.0 - np.mean(np.abs(img - flipped_ud))

    features = [f1_area, f2_perimeter, f3_thickness, f4_major_axis, f5_eccentricity, 
                f6_orientation, f7_solidity, f8_extent, f9_aspect_ratio, 
                f10_euler, f11_h_symmetry, f12_v_symmetry]
    return torch.tensor(features, dtype=torch.float32)

class MorphMNIST12(Dataset):
    """MNIST 데이터셋을 로드하고 형태학적 특징(M)을 미리 계산하여 캐싱"""
    def __init__(self, train=True, limit_count=None): 
        self.mnist = datasets.MNIST(root='../data', train=train, download=True, 
                                    transform=transforms.ToTensor())
        
        if limit_count is not None and limit_count < len(self.mnist):
            print(f"[Info] 전체 데이터 중 {limit_count}개만 사용")
            self.indices = range(limit_count)
        else:
            print(f"[Info] 전체 데이터({len(self.mnist)}개) 사용")
            self.indices = range(len(self.mnist))

        self.cached_data = []
        print(f"Pre-computing features for {'Train' if train else 'Test'} set...")
        
        for i, idx in enumerate(self.indices):
            img, label = self.mnist[idx]
            m = extract_12_features(img)
            t_onehot = torch.zeros(10)
            t_onehot[label] = 1.0
            self.cached_data.append((img, m, t_onehot))
            if (i + 1) % 500 == 0:
                print(f"Processing... {i + 1}/{len(self.indices)}", end='\r')
        
        print(f"\nFeature extraction complete")

    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]
    