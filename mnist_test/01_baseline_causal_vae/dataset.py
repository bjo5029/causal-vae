import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from skimage.measure import label as sk_label, regionprops 
from scipy.ndimage import distance_transform_edt

from skimage.morphology import skeletonize
import cv2

def extract_refined_features(img_tensor):
    """
    이미지에서 12가지 정제된 형태학적(Morphological) 특징 추출
    1. Area
    2. Perimeter
    3. Thickness
    4. Major Axis Length
    5. Eccentricity
    6. Orientation
    7. Solidity
    8. Extent
    9. Aspect Ratio
    10. Euler Number
    11. Horizontal Symmetry
    12. Vertical Symmetry
    """
    img = img_tensor.squeeze().numpy()
    binary = img > 0.2
    
    if np.sum(binary) == 0: return torch.zeros(12, dtype=torch.float32)

    labeled_img = sk_label(binary)
    props = regionprops(labeled_img)
    
    if not props: return torch.zeros(12, dtype=torch.float32)
    
    # 가장 큰 덩어리만 선택 (노이즈 제거)
    prop = props[np.argmax([p.area for p in props])]
    
    # 1. Area
    f1_area = prop.area / 784.0
    
    # 2. Perimeter
    f2_perimeter = prop.perimeter / 100.0
    
    # 3. Thickness
    dist_map = distance_transform_edt(binary)
    f3_thickness = np.max(dist_map) / 5.0
    
    # 4. Major Axis Length
    f4_major_axis = prop.major_axis_length / 28.0
    
    # 5. Eccentricity
    f5_eccentricity = prop.eccentricity
    
    # 6. Orientation
    # Normalize -pi/2 to pi/2 range to 0 to 1
    f6_orientation = (prop.orientation + np.pi/2) / np.pi
    
    # 7. Solidity
    f7_solidity = prop.solidity
    
    # 8. Extent
    f8_extent = prop.extent
    
    # 9. Aspect Ratio
    minr, minc, maxr, maxc = prop.bbox
    height = maxr - minr
    width = maxc - minc
    
    # Height가 0일 경우 예외 처리
    if height == 0: 
        f9_aspect_ratio = 0.0
    else:
        # 가로/세로 비율 (1.0 기준)
        # 단순히 width/height로 하면 범위가 너무 커질 수 있으므로 조정
        ratio = width / height
        f9_aspect_ratio = ratio / 3.0 # 대략 0~1 범위로
    
    # 10. Euler Number
    # MNIST는 보통 구멍이 없거나(1), 1개(0), 2개(-1)
    # 정규화: (euler + 2) / 4.0
    f10_euler = (prop.euler_number + 2) / 4.0 
    
    # 11. Horizontal Symmetry
    flipped_lr = np.fliplr(img)
    f11_h_symmetry = 1.0 - np.mean(np.abs(img - flipped_lr))
    
    # 12. Vertical Symmetry
    flipped_ud = np.flipud(img)
    f12_v_symmetry = 1.0 - np.mean(np.abs(img - flipped_ud))

    features = [
        f1_area, f2_perimeter, f3_thickness, f4_major_axis, f5_eccentricity,
        f6_orientation, f7_solidity, f8_extent, f9_aspect_ratio, f10_euler,
        f11_h_symmetry, f12_v_symmetry
    ]
                
    return torch.tensor(features, dtype=torch.float32)

class MorphMNIST12(Dataset):
    """MNIST 데이터셋을 로드하고 M을 미리 계산해서 캐싱"""
    def __init__(self, train=True, limit_count=None): 
        self.mnist = datasets.MNIST(root='../../data', train=train, download=True, 
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
            m = extract_refined_features(img)
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
    