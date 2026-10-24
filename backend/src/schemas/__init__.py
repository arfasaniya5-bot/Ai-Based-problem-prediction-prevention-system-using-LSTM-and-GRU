"""Pydantic schemas package for DeviceGuard AI."""

from .health import HealthResponse, ModelHealthResponse
from .prediction import (
    DailyTelemetryInput,
    PredictionRequest,
    PredictionResponse,
    PredictionRecordResponse,
)
from .device import DeviceResponse, PaginatedDevicesResponse
from .alert import AlertResponse, PaginatedAlertsResponse, AlertAcknowledgeResponse

__all__ = [
    "HealthResponse",
    "ModelHealthResponse",
    "DailyTelemetryInput",
    "PredictionRequest",
    "PredictionResponse",
    "PredictionRecordResponse",
    "DeviceResponse",
    "PaginatedDevicesResponse",
    "AlertResponse",
    "PaginatedAlertsResponse",
    "AlertAcknowledgeResponse",
]
