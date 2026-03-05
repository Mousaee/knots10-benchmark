#!/usr/bin/env python3
"""Grad-CAM visualization for knot classification models."""
import os, json, glob, random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
RESULTS_DIR = 'results'
os.makedirs(f'{RESULTS_DIR}/figures/gradcam', exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class GradCAM:
    """Grad-CAM implementation for CNN models."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, gin, gout):
            self.gradients = gout[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)
        
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class

def overlay_cam(img_pil, cam, alpha=0.5):
    """Overlay CAM heatmap on original image."""
    from PIL import Image as PILImage
    import cv2
    img_np = np.array(img_pil.resize((224,224)))
    h, w = img_np.shape[:2]
    cam_resized = np.array(
        PILImage.fromarray((cam*255).astype(np.uint8)).resize((w,h)))
    cam_resized = cam_resized.astype(np.float32) / 255.0
    heatmap = cm.jet(cam_resized)[:,:,:3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1-alpha) * img_np).astype(np.uint8)
    return overlay

def get_sample_images(data_dir, n_per_class=2):
    """Get sample test images for each class."""
    samples = {}
    for cls in CLASSES:
        pattern = os.path.join(data_dir, '**', f'{cls}_*Set*.jpg')
        files = glob.glob(pattern, recursive=True)
        if files:
            samples[cls] = random.sample(files, min(n_per_class, len(files)))
    return samples

def run_gradcam_resnet18(data_dir):
    """Run Grad-CAM on ResNet-18."""
    device = torch.device('cpu')  # CPU for viz
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(CLASSES)))
    
    # Load best weights if saved, else use pretrained
    ckpt = f'{RESULTS_DIR}/resnet18_best.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        # Re-init with pretrained and fine-tune head
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, len(CLASSES)))
        print("[WARN] No checkpoint found, using pretrained weights")
    
    model.eval()
    target_layer = model.layer4[-1]
    return model, target_layer, device
