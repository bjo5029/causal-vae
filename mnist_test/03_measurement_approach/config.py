import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CONFIG = {
    "BATCH_SIZE": 128,
    "EPOCHS": 30,
    "LR": 1e-3,
    "Z_DIM": 10,
    "M_DIM": 16, 
    "T_DIM": 10,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SEED": 42,
    "BETA": 1.0, # Reduced from 5.0 to 1.0 to improve reconstruction
    "LAMBDA_ADV": 10.0 # Weight for adversarial loss (disentanglement)
}

FEATURE_NAMES = [
    "Area", "Thickness", "Solidity", "AspectRatio", "Euler", 
    "H_Symmetry", "V_Symmetry", "Endpoints", "Junctions",
    "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"
]
