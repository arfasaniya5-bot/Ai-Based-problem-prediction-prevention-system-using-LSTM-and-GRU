"""Application configuration settings for DeviceGuard AI."""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "DeviceGuard AI Backend"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    API_PREFIX: str = "/api"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'deviceguard.db'}")

    # Model & Inference Defaults
    DEFAULT_MODEL_NAME: str = "LSTM Neural Network"
    DECISION_THRESHOLD: float = 0.50
    SEQUENCE_LENGTH: int = 3
    HORIZON_DAYS: int = 2

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
