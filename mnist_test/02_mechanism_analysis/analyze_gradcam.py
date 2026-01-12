
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import CONFIG
from dataset import MorphMNIST12
from models import CausalMorphVAE12, SimpleClassifier
from train import train_model

# ---------------------------------------------------------
# 1. Grad-CAM Helper Class
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook registration
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        if isinstance(output, tuple):
             # SimpleClassifier returns (features, log_softmax)
             logits = output[1]
        else:
             logits = output
             
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        # Generator Heatmap
        gradients = self.gradients.data.cpu().numpy()[0] # (Channels, H, W)
        activations = self.activations.data.cpu().numpy()[0] # (Channels, H, W)
        
        # Global Average Pooling of Gradients (Weights)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, class_idx

# ---------------------------------------------------------
# 2. Re-using Logic from Residual Analysis
# ---------------------------------------------------------
def get_residuals_and_train_classifier(device):
    # (A) Train or Load VAE
    print("[Grad-CAM] Training/Loading VAE...")
    vae = train_model()
    vae.eval()
    
    # (B) Get Residuals
    print("[Grad-CAM] Generating Residuals...")
    full_dataset = MorphMNIST12(train=False, limit_count=5000) # Use subset for speed
    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    
    residuals_list = []
    labels_list = []
    
    with torch.no_grad():
        for x, m, t in loader:
            x, m, t = x.to(device), m.to(device), t.to(device)
            recon_x, _, _, _ = vae(x, m, t)
            recon_x = recon_x.view(-1, 1, 28, 28)
            
            # RESIDUAL: |X - X_hat|
            # Using absolute difference to see "where" the error is clearly
            res = torch.abs(x - recon_x)
            
            residuals_list.append(res.cpu())
            labels_list.append(torch.argmax(t, dim=1).cpu())
            
    residuals = torch.cat(residuals_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # (C) Train Classifier on Residuals
    print("[Grad-CAM] Training Classifier on Residuals...")
    dataset = TensorDataset(residuals, labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    classifier = SimpleClassifier().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    classifier.train()
    for epoch in range(5): # Train enough to get high accuracy
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            _, out = classifier(bx)
            loss = F.nll_loss(out, by)
            loss.backward()
            optimizer.step()
            
    print("[Grad-CAM] Classifier Ready.")
    return classifier, residuals, labels

# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
def main():
    device = CONFIG["DEVICE"]
    
    # 1. Prepare Models and Data
    classifier, residuals, labels = get_residuals_and_train_classifier(device)
    
    # 2. Setup Grad-CAM
    # SimpleClassifier has self.conv2 as the last conv layer
    grad_cam = GradCAM(classifier, classifier.conv2)
    
    # 3. Select samples (1 per digit) to visualize
    print("[Grad-CAM] Visualizing Heatmaps...")
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    found_digits = {}
    
    indices = np.random.permutation(len(residuals))
    
    for idx in indices:
        res_img = residuals[idx].unsqueeze(0).to(device) # (1, 1, 28, 28)
        label = labels[idx].item()
        
        if label in found_digits:
            continue
            
        # Get CAM
        heatmap, pred_idx = grad_cam(res_img, class_idx=label)
        
        # Plot Residual Image
        ax_res = axes[0, label]
        ax_res.imshow(res_img.cpu().squeeze(), cmap='gray')
        ax_res.axis('off')
        if label == 0:
            ax_res.set_ylabel("Residual", fontsize=12, fontweight='bold')
        ax_res.set_title(f"Digit {label}", fontsize=12)
        
        # Plot Heatmap Overlay
        ax_cam = axes[1, label]
        ax_cam.imshow(res_img.cpu().squeeze(), cmap='gray')
        ax_cam.imshow(heatmap, cmap='jet', alpha=0.5) # Overlay
        ax_cam.axis('off')
        if label == 0:
            ax_cam.set_ylabel("Grad-CAM", fontsize=12, fontweight='bold')
            
        found_digits[label] = True
        if len(found_digits) == 10:
            break
            
    plt.suptitle(f"Why did the classifier predict the digit from Residuals?", fontsize=16)
    save_path = "residual_gradcam_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"[Done] Result saved to {save_path}")

if __name__ == "__main__":
    main()
