"""API package for DeviceGuard AI."""

from .routes_health import router as health_router
from .routes_prediction import router as prediction_router

__all__ = ["health_router", "prediction_router"]
