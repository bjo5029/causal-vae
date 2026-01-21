import torch

class Config:
    def __init__(self):
        # 1. Device
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Hyperparameters
        self.EPOCHS = 100
        self.BATCH_SIZE = 8
        self.LEARNING_RATE = 1e-4
        self.BETA = 0.0001      # KLD Weight (Almost AE)
        self.LAMBDA_ADV = 0.0   # Adversarial Loss Weight (Disabled for stability)
        
        # 3. Data Dimensions
        self.IMG_HEIGHT = 768  # 768 = 128 * 6
        self.IMG_WIDTH = 1280  # 1280 = 128 * 10
        # self.IMG_SIZE = (512, 1024) # (H, W) tuple if needed
        
        self.T_DIM = 19      # 19 Treatment Groups
        self.M_DIM = 12      # 12 Morphological Features
        self.Z_DIM = 512      
        
        # 4. Paths
        self.DATA_CSV = "../../data/vessel_analysis_result.csv"
        self.DATA_ROOT = "../../data/mip_imgs_plates_25250-25254" # Where Plate-* folders are
        self.SAVE_DIR = "saved_models"
        self.RESULT_DIR = "results"

CONFIG = Config().__dict__
