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
    Load model weights from checkpoint if available, otherwise use
    pretrained ImageNet backbone with an untrained classification head.
    Returns the model in eval mode.
    """
    global _model, _model_metadata

    if _model is not None:
        return _model

    model = _build_efficientnet()
    checkpoint_path = settings.model_path

    if os.path.isfile(checkpoint_path):
        log.info("Loading trained weights from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        _model_metadata = {
            "version": checkpoint.get("model_version", settings.model_version),
            "training_date": checkpoint.get("training_date"),
            "metrics": checkpoint.get("metrics"),
        }
        log.info("Model loaded successfully (version %s)", _model_metadata["version"])
    else:
        log.warning(
            "No trained checkpoint at %s — using pretrained ImageNet backbone with untrained head. "
            "Predictions will be unreliable until a trained model is provided.",
            checkpoint_path,
        )
        _model_metadata = {
            "version": settings.model_version + "-pretrained",
            "training_date": None,
            "metrics": None,
        }

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
