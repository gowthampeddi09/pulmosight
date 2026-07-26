import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

_model: Optional[nn.Module] = None
_model_metadata: dict = {}


def _build_efficientnet() -> nn.Module:
    """Build EfficientNet-B0 with a binary classification head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace classifier: 1280 features -> 1 output (sigmoid for binary)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1),
    )
    return model


def load_model() -> nn.Module:
    """
    Load fine-tuned model weights from checkpoint.
    Searches multiple path locations across Docker and local execution contexts.
    """
    global _model, _model_metadata

    if _model is not None:
        return _model

    model = _build_efficientnet()
    
    # Candidate checkpoint locations across docker, backend, and root execution contexts
    base_dir = Path(__file__).resolve().parent.parent.parent
    candidate_paths = [
        "/app/weights/best_model.pth",
        "weights/best_model.pth",
        "backend/weights/best_model.pth",
        settings.model_path,
        str(base_dir / "weights" / "best_model.pth"),
        str(base_dir.parent / "weights" / "best_model.pth"),
    ]

    checkpoint_path = None
    for path in candidate_paths:
        if os.path.isfile(path):
            checkpoint_path = path
            break

    if checkpoint_path:
        log.info("Loading fine-tuned PyTorch weights from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        _model_metadata = {
            "version": checkpoint.get("model_version", settings.model_version),
            "training_date": checkpoint.get("training_date"),
            "metrics": checkpoint.get("metrics"),
        }
        log.info("Fine-tuned PyTorch Model loaded successfully (version %s)", _model_metadata["version"])
    else:
        log.error("CRITICAL ERROR: No trained checkpoint found in candidate locations: %s", candidate_paths)
        raise FileNotFoundError(f"Fine-tuned PyTorch model weights missing from candidate paths: {candidate_paths}")

    model.eval()
    _model = model
    return _model


def get_model_metadata() -> dict:
    """Return model metadata. Loads model first if not already loaded."""
    if _model is None:
        load_model()
    return _model_metadata


def is_model_loaded() -> bool:
    return _model is not None


def get_target_layer(model: nn.Module):
    """Return the last convolutional layer of EfficientNet-B0 for Grad-CAM."""
    return model.features[-1]
