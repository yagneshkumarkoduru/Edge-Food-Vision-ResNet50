"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ResNet-50
Author: Yagnesh Kumar Koduru
Repository: Food-Classification-Using-ResNet-50
Domain: Explainable AI (XAI), Computer Vision, Edge Intelligence

Generates visual attention heatmaps indicating which spatial regions
in a food image activate the ResNet-50 bottleneck feature representations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM for a PyTorch convolutional model.
        
        Args:
            model: PyTorch model (e.g. ResNet-50)
            target_layer: Convolutional layer to inspect (e.g. model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a given input tensor.
        
        Args:
            input_tensor: (1, 3, H, W) normalized image tensor
            target_class: int, class index to explain (if None, uses predicted class)
            
        Returns:
            cam_map: (H, W) numpy array normalized to [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Zero gradients and backpropagate target class score
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients: alpha_k = 1/Z sum_i sum_j (d y_c / d A_k_ij)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU to retain features with positive influence on class c
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class

    def overlay_heatmap(self, original_img_np, cam_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Superimpose Grad-CAM heatmap over original RGB image.
        
        Args:
            original_img_np: (H, W, 3) uint8 numpy array [0, 255]
            cam_map: (H, W) float numpy array [0, 1]
            alpha: blending ratio
            
        Returns:
            blended_img: (H, W, 3) uint8 RGB visualization
        """
        heatmap = (cam_map * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = (alpha * heatmap_rgb + (1.0 - alpha) * original_img_np).astype(np.uint8)
        return blended
