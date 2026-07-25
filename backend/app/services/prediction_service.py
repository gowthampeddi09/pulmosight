import uuid
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.inference.predictor import predict, get_preprocessed_tensor
from app.inference.model_loader import load_model, get_target_layer
from app.xai.gradcam import GradCAM
from app.xai.overlay import create_overlay
from app.xai.observation import generate_observation
from app.utils.file_handling import save_upload, save_heatmap
from app.models.prediction import Prediction

log = logging.getLogger(__name__)


async def run_prediction(
    db: AsyncSession,
    user_id: uuid.UUID,
    image_bytes: bytes,
    original_filename: str,
) -> Prediction:
    """
    Full prediction pipeline:
    validate image -> save file -> inference -> Grad-CAM -> save heatmap -> persist to DB.
    """
    # 1. Save the uploaded image to disk
    unique_filename, image_path = save_upload(image_bytes, original_filename)

    # 2. Run model inference
    result = predict(image_bytes)

    # 3. Grad-CAM heatmap generation
    heatmap_path = None
    observation_text = None
    try:
        model = load_model()
        target_layer = get_target_layer(model)
        grad_cam = GradCAM(model, target_layer)

        tensor = get_preprocessed_tensor(image_bytes)
        cam = grad_cam.generate(tensor)
        grad_cam.cleanup()

        overlay_bytes = create_overlay(image_bytes, cam)
        prediction_id = uuid.uuid4()
        heatmap_path = save_heatmap(overlay_bytes, str(prediction_id))
        observation_text = generate_observation(cam)
    except Exception as e:
        log.error("Grad-CAM failed (non-fatal): %s", e)
        prediction_id = uuid.uuid4()

    # 4. Persist to database
    prediction = Prediction(
        id=prediction_id,
        user_id=user_id,
        filename=unique_filename,
        original_image_path=image_path,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        processing_time_ms=result["processing_time_ms"],
        heatmap_path=heatmap_path,
        gradcam_observation=observation_text,
    )
    db.add(prediction)
    await db.flush()
    await db.refresh(prediction)

    log.info(
        "Prediction saved: id=%s label=%s confidence=%.4f",
        prediction.id, prediction.prediction, prediction.confidence,
    )
    return prediction
