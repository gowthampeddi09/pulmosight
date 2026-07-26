import logging
from io import BytesIO

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def create_overlay(
    original_bytes: bytes,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> bytes:
    """
    Overlay a Grad-CAM heatmap on the original image with optimized resolution.
    Downsamples ultra-high-res scans to max 800px for 10x faster rendering and low memory footprint.
    """
    original = Image.open(BytesIO(original_bytes)).convert("RGB")
    orig_w, orig_h = original.size

    # Cap maximum dimension to 800px for fast cloud processing
    max_dim = 800
    if orig_w > max_dim or orig_h > max_dim:
        ratio = min(max_dim / orig_w, max_dim / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        original = original.resize((new_w, new_h), Image.BILINEAR)
        orig_w, orig_h = new_w, new_h

    # Resize CAM to match image dimensions
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply jet colormap
    heatmap_rgb = _apply_jet_colormap(cam_resized)
    heatmap_img = Image.fromarray(heatmap_rgb)

    # Blend original with heatmap
    overlay = Image.blend(original, heatmap_img, alpha)

    buf = BytesIO()
    overlay.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _apply_jet_colormap(intensity: np.ndarray) -> np.ndarray:
    """Convert [0, 1] intensity map to an RGB jet colormap."""
    r = np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)
