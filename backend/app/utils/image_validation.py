import logging
from io import BytesIO
from PIL import Image
from fastapi import UploadFile, HTTPException, status

from app.schemas.common import make_error
from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def validate_upload(file: UploadFile) -> bytes:
    """
    Validate an uploaded file is a real, non-corrupted image of acceptable
    type and resolution. Returns the raw bytes on success.
    """
    # Check content type header
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("INVALID_FILE_TYPE", f"Only JPEG and PNG images are accepted. Got: {file.content_type}"),
        )

    # Check extension
    filename = (file.filename or "").lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("INVALID_FILE_TYPE", "File extension must be .jpg, .jpeg, or .png"),
        )

    raw = await file.read()

    # Size check
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("FILE_TOO_LARGE", f"Maximum file size is {settings.max_file_size_mb}MB"),
        )

    # Try opening with Pillow — catches corrupted files
    try:
        img = Image.open(BytesIO(raw))
        img.verify()  # actually checks integrity
        # Re-open because verify() can close the file pointer
        img = Image.open(BytesIO(raw))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error("CORRUPTED_IMAGE", "The uploaded file is not a valid image or is corrupted"),
        )

    w, h = img.size
    if w < settings.min_image_dimension or h < settings.min_image_dimension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=make_error(
                "LOW_RESOLUTION",
                f"Image must be at least {settings.min_image_dimension}x{settings.min_image_dimension}px. Got {w}x{h}",
            ),
        )

    log.info("Validated image upload: %s (%dx%d, %d bytes)", filename, w, h, len(raw))
    return raw
