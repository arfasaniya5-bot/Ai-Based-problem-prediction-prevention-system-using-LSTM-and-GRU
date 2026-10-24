"""Services package for DeviceGuard AI."""

from .inference_service import ModelInferenceService, get_inference_service

__all__ = ["ModelInferenceService", "get_inference_service"]
