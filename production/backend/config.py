"""Application settings loaded from environment variables prefixed with DM_."""
from pathlib import Path
from pydantic_settings import BaseSettings

_REPO_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Pydantic settings model; all fields overridable via DM_* env vars."""

    data_raw_dir: Path = _REPO_ROOT / "data/raw"
    data_processed_dir: Path = _REPO_ROOT / "data/processed"
    models_dir: Path = _REPO_ROOT / "models"

    cors_origins: list[str] = ["*"]

    max_workers: int = 2

    class Config:
        env_prefix = "DM_"


settings = Settings()
