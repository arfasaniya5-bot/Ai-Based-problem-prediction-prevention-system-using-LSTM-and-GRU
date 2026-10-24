"""Health and system status schemas."""

from datetime import datetime, timezone
from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Basic service health status."""

    status: str = Field(..., examples=["healthy"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field(..., examples=["1.0.0"])
    environment: str = Field(..., examples=["development"])


class ModelHealthResponse(BaseModel):
    """Model artifacts loading health status."""

    status: str = Field(..., examples=["loaded"])
    models_loaded: Dict[str, bool] = Field(
        ...,
        examples=[{"lstm": True, "gru": True, "baseline": True}],
    )
    feature_count: int = Field(..., examples=[11])
    sequence_length: int = Field(..., examples=[3])
    horizon_days: int = Field(..., examples=[2])
    decision_threshold: float = Field(..., examples=[0.50])
    device: str = Field(..., examples=["cpu"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
