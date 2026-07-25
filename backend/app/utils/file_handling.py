import os
import uuid
import logging
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


def ensure_dirs() -> None:
    """Create upload directories if they don't exist."""
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.heatmap_dir).mkdir(parents=True, exist_ok=True)


def save_upload(raw_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """
    Save uploaded image bytes to disk with a unique filename.
    Returns (unique_filename, full_path).
    """
    ensure_dirs()
    ext = Path(original_filename).suffix.lower() or ".png"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    full_path = os.path.join(settings.upload_dir, unique_name)
    with open(full_path, "wb") as f:
        f.write(raw_bytes)
    log.info("Saved upload: %s -> %s", original_filename, full_path)
    return unique_name, full_path


def save_heatmap(image_bytes: bytes, prediction_id: str) -> str:
    """Save a Grad-CAM heatmap overlay to the heatmaps directory."""
    ensure_dirs()
    filename = f"{prediction_id}_heatmap.png"
    full_path = os.path.join(settings.heatmap_dir, filename)
    with open(full_path, "wb") as f:
        f.write(image_bytes)
    return full_path


def delete_prediction_files(image_path: str | None, heatmap_path: str | None) -> None:
    """Remove image and heatmap files from disk. Ignores missing files."""
    for path in (image_path, heatmap_path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
                log.info("Deleted file: %s", path)
            except OSError as e:
                log.warning("Failed to delete %s: %s", path, e)
