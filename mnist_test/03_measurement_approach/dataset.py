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
    이미지에서 16가지 정제된 형태학적(Morphological) 특징 추출
    Removed: Perimeter, MajorAxis, Eccentricity, Orientation, Extent
    Added: Skeleton Endpoints, Skeleton Junctions, Hu Moments (7)
    """
    img = img_tensor.squeeze().numpy()
    binary = img > 0.2
    
    if np.sum(binary) == 0: return torch.zeros(16, dtype=torch.float32)

    labeled_img = sk_label(binary)
    props = regionprops(labeled_img)
    
    if not props: return torch.zeros(16, dtype=torch.float32)
    
    # 가장 큰 덩어리만 선택 (노이즈 제거)
    prop = props[np.argmax([p.area for p in props])]

    # 1. 기본 기하학적 특징 (5개)
    f1_area = prop.area / 784.0
    dist_map = distance_transform_edt(binary)
    f2_thickness = np.max(dist_map) / 5.0
    f3_solidity = prop.solidity
    
    minr, minc, maxr, maxc = prop.bbox
    height = maxr - minr
    width = maxc - minc
    f4_aspect_ratio = (width / height) if height > 0 else 0
    f4_aspect_ratio = np.clip(f4_aspect_ratio, 0, 3.0) / 3.0 
    
    # 위상 (Topology)
    f5_euler = (prop.euler_number + 2) / 4.0 
    
    # 2. 대칭성 특징 (2개)
    flipped_lr = np.fliplr(img)
    f6_h_symmetry = 1.0 - np.mean(np.abs(img - flipped_lr))
    flipped_ud = np.flipud(img)
    f7_v_symmetry = 1.0 - np.mean(np.abs(img - flipped_ud))

    # 3. Skeleton 특징 (2개) - 획의 구조
    # Skeletonize
    skel = skeletonize(binary)
    skel_int = skel.astype(np.uint8)
    
    # Endpoints & Junctions detection using hit-or-miss transform or simple kernel convolution
    # Simple method: Neighboar count
    # 3x3 kernel with 1 at center and 1s neighbors.
    # Check connectivity.
    # More robust: use dedicated lookup table or simpler summation for 1-pixel skeleton
    
    # Filter 3x3 neighbor sum where center is 1
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    filtered = cv2.filter2D(skel_int, -1, kernel)
    
    # Center pixel = 10. Neighbors contribute 1 each.
    # Endpoint: Sum = 11 (Center 10 + 1 neighbor)
    # Junction: Sum >= 13 (Center 10 + 3 or more neighbors)
    # Note: skeleton endpoints might have 1 neighbor.
    
    endpoints = np.sum(filtered == 11)
    junctions = np.sum(filtered >= 13)
    
    f8_endpoints = endpoints / 5.0  # Normalize roughly
    f9_junctions = junctions / 5.0

    # 4. Hu Moments (7개) - 전반적인 형상
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log scale transform for stability: -1 * sign * log10(abs)
    # But for simplicity in VAE, raw or simple transform.
    # Hu moments range drastically. Often log transform is used.
    # Here we just take raw or standardized? Raw can be 1e-3 to 1e-7. 
    # Let's apply log transform: -log10(|h|)
    
    hu_feats = []
    for h in hu_moments:
        val = -1.0 * np.sign(h) * np.log10(np.abs(h) + 1e-10)
        # Normalize roughly to 0-1 range? It's usually around 0~30?
        val = val / 10.0 
        hu_feats.append(val)
        
    features = [f1_area, f2_thickness, f3_solidity, f4_aspect_ratio, f5_euler,
                f6_h_symmetry, f7_v_symmetry, f8_endpoints, f9_junctions] + hu_feats
                
    return torch.tensor(features, dtype=torch.float32)

class MorphMNIST12(Dataset):
    """MNIST 데이터셋을 로드하고 형태학적 특징(M)을 미리 계산하여 캐싱"""
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
    