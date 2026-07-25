"""
Hand-rolled Grad-CAM implementation on EfficientNet-B0.

Using a custom implementation instead of pytorch-grad-cam to demonstrate
understanding of the technique and avoid an extra dependency.
"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

log = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer, runs a forward+backward pass,
    then computes a weighted combination of feature maps to produce a
    class-discriminative heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks — these capture activations and gradients during forward/backward
        self._fwd_hook = target_layer.register_forward_hook(self._store_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._store_gradient)

    def _store_activation(self, module, inp, output):
        self.activations = output.detach()

    def _store_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.
        Returns a 2D numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()  # keep BN/dropout in eval mode
        # We still need gradients for the hooks, so use enable_grad context

        output = self.model(input_tensor)
        # Binary classification: single logit output
        score = torch.sigmoid(output) if output.shape[-1] == 1 else output.max()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            log.error("Grad-CAM hooks did not fire — check target layer")
            # Return uniform heatmap as fallback
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Global average pooling of gradients -> channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)  # Only positive contributions

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
