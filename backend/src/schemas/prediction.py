"""Inference and telemetry request/response schemas."""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DailyTelemetryInput(BaseModel):
    """Single-day observation of 11 verified SMART raw features."""

    smart_1_raw: Optional[float] = Field(None, description="Raw Read Error Rate")
    smart_4_raw: Optional[float] = Field(None, description="Start/Stop Count")
    smart_5_raw: Optional[float] = Field(None, description="Reallocated Sectors Count")
    smart_7_raw: Optional[float] = Field(None, description="Seek Error Rate")
    smart_9_raw: Optional[float] = Field(None, description="Power-On Hours")
    smart_12_raw: Optional[float] = Field(None, description="Power Cycle Count")
    smart_187_raw: Optional[float] = Field(None, description="Reported Uncorrectable Errors")
    smart_188_raw: Optional[float] = Field(None, description="Command Timeout")
    smart_194_raw: Optional[float] = Field(None, description="Temperature Celsius")
    smart_197_raw: Optional[float] = Field(None, description="Current Pending Sector Count")
    smart_198_raw: Optional[float] = Field(None, description="Offline Uncorrectable Sector Count")

    @field_validator("*", mode="before")
    def validate_numeric(cls, v):
        if v is None:
            return None
        try:
            val = float(v)
            if not (-1e15 < val < 1e15):
                raise ValueError("Value out of valid numerical range")
            return val
        except (TypeError, ValueError):
            raise ValueError("Telemetry feature values must be numeric numbers")


class PredictionRequest(BaseModel):
    """Inference request containing a sequence of 3 consecutive daily telemetry readings."""

    serial_number: Optional[str] = Field(None, max_length=64, examples=["W30068RN"], description="Hard drive serial number")
    model: Optional[str] = Field(None, max_length=64, examples=["ST4000DM000"], description="Hard drive model string")
    readings: List[DailyTelemetryInput] = Field(
        ...,
        description="Exactly 3 consecutive days of SMART readings [Day t-2, Day t-1, Day t]",
    )

    @field_validator("readings")
    def validate_sequence_length(cls, v):
        if len(v) != 3:
            raise ValueError(f"Inference requires exactly 3 consecutive daily readings (received {len(v)})")
        return v


class PredictionResponse(BaseModel):
    """Inference response returning probability score, prediction class, and alert recommendation."""

    serial_number: Optional[str] = Field(None, examples=["W30068RN"])
    model_name: str = Field(..., examples=["LSTM Neural Network"])
    model_version: str = Field(..., examples=["1.0.0"])
    prediction_class: int = Field(..., examples=[1], description="0: Healthy, 1: Failure Predicted")
    is_failure_predicted: bool = Field(..., examples=[True])
    risk_score: float = Field(..., examples=[0.8421], description="Model sigmoid failure probability [0.0, 1.0]")
    risk_level: str = Field(..., examples=["CRITICAL"], description="LOW, MODERATE, CRITICAL")
    threshold: float = Field(..., examples=[0.50], description="Decision threshold applied")
    horizon_days: int = Field(..., examples=[2], description="Advance warning horizon (days)")
    sequence_length_days: int = Field(..., examples=[3])
    recommendation: str = Field(..., description="Actionable preventative recommendation")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    disclaimer: str = Field(
        "Academic research prototype; not certified for production deployment.",
        description="Safety and readiness disclosure",
    )


class PredictionRecordResponse(BaseModel):
    """Stored prediction history record."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    device_id: Optional[int]
    timestamp: datetime
    model_name: str
    prediction_class: int
    risk_score: float
    risk_level: str
    threshold: float
    horizon_days: int
    status: str
