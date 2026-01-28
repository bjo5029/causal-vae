import torch

class Config:
    def __init__(self):
        # 1. Device
        self.GPU_ID = 0 # Set to 0, 1, 2 etc.
        self.DEVICE = torch.device(f"cuda:{self.GPU_ID}" if torch.cuda.is_available() else "cpu")
        
        # 2. Hyperparameters
        self.EPOCHS = 150
        self.BATCH_SIZE = 8
        self.LEARNING_RATE = 1e-4
        self.BETA = 0.5     # KLD Weight (Almost AE -> Weak VAE)
        self.LAMBDA_MORPH = 10000  # Morphology Prediction Loss Weight
        
        # 3. Data Dimensions
        self.IMG_HEIGHT = 768  # 768 = 128 * 6
        self.IMG_WIDTH = 1280  # 1280 = 128 * 10
        # self.IMG_SIZE = (512, 1024) # (H, W) tuple if needed
        
        self.T_DIM = 19      # 19 Treatment Groups
        self.M_DIM = 12      # 12 Morphological Features
        self.Z_DIM = 128      
        
        # 4. Paths
        # 4. Paths
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Project root: 00_core -> 01_causal_vae -> vessel_analysis -> workspace line (so ../../../)
        project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
        
        self.DATA_CSV = os.path.join(project_root, "data/vessel_analysis_result.csv")
        self.DATA_ROOT = os.path.join(project_root, "data/mip_imgs_plates_25250-25254")
        
        # Output dirs 
        self.SAVE_DIR = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_saved_models_kfold_morph10000"
        self.RESULT_DIR = "/home/jeongeun.baek/workspace/causal-vae/vessel_analysis/01_causal_vae/outputs/7_results_kfold_morph10000"

CONFIG = Config().__dict__
