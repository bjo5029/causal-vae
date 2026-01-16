import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CONFIG = {
    "BATCH_SIZE": 128,
    "EPOCHS": 30,
    "LR": 1e-3,
    "Z_DIM": 10,
    "M_DIM": 12, 
    "T_DIM": 10,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SEED": 42,
    "BETA": 1.0, # Reduced from 5.0 to 1.0 to improve reconstruction
    "LAMBDA_ADV": 10.0 # Weight for adversarial loss (disentanglement)
}

FEATURE_NAMES = [
    "Area", "Perimeter", "Thickness", "MajorAxis", "Eccentricity", 
    "Orientation", "Solidity", "Extent", "AspectRatio", "Euler", 
    "H_Symmetry", "V_Symmetry"
]
