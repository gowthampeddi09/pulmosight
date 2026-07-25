import time
import logging
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from app.inference.model_loader import load_model, get_model_metadata

log = logging.getLogger(__name__)

# ImageNet normalization — must match training pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image_bytes: bytes) -> dict:
    """
    Run pneumonia/normal inference on raw image bytes.
    Returns dict with prediction label, confidence, processing time, and model version.
    """
    model = load_model()
    metadata = get_model_metadata()

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]

    start = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor)  # [1, 1]
        prob = torch.sigmoid(logit).item()
    elapsed_ms = (time.perf_counter() - start) * 1000

    label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if label == "PNEUMONIA" else 1.0 - prob

    log.info(
        "Prediction: %s (confidence=%.4f, time=%.1fms, version=%s)",
        label, confidence, elapsed_ms, metadata["version"],
    )

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "processing_time_ms": round(elapsed_ms, 2),
        "model_version": metadata["version"],
    }


def get_preprocessed_tensor(image_bytes: bytes) -> torch.Tensor:
    """Return preprocessed tensor for Grad-CAM (keeps grad computation graph)."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    tensor.requires_grad_(True)
    return tensor
