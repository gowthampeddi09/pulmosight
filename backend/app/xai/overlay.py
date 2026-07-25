import logging
from io import BytesIO

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def create_overlay(
    original_bytes: bytes,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> bytes:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_bytes: Raw bytes of the original uploaded image.
        cam: 2D numpy array (H, W) with values in [0, 1].
        alpha: Blending factor. Higher = more heatmap visibility.
        colormap_name: Matplotlib colormap name for the heatmap.

    Returns:
        PNG image bytes of the overlay.
    """
    original = Image.open(BytesIO(original_bytes)).convert("RGB")
    orig_w, orig_h = original.size

    # Resize CAM to match original image dimensions
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply a simple jet-like colormap without matplotlib dependency
    heatmap_rgb = _apply_jet_colormap(cam_resized)
    heatmap_img = Image.fromarray(heatmap_rgb)

    # Blend original with heatmap
    overlay = Image.blend(original, heatmap_img, alpha)

    buf = BytesIO()
    overlay.save(buf, format="PNG")
    return buf.getvalue()


def _apply_jet_colormap(intensity: np.ndarray) -> np.ndarray:
    """
    Convert a [0, 1] intensity map to an RGB jet colormap.
    Avoids pulling in matplotlib just for a colormap.
    """
    # Jet colormap approximation: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)
