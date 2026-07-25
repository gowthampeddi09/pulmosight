from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "PulmoSight"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://pulmosight:pulmosight_secret@postgres:5432/pulmosight"

    # Auth
    secret_key: str = "change-me"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # LLM
    google_api_key: str = ""
    groq_api_key: str = ""

    # Model
    model_path: str = "weights/best_model.pth"
    model_version: str = "1.0.0"

    # File storage
    upload_dir: str = "uploads"
    heatmap_dir: str = "uploads/heatmaps"
    max_file_size_mb: int = 10
    min_image_dimension: int = 100

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
