"""Health check routes for DeviceGuard AI."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from src.config import settings
from src.schemas.health import HealthResponse, ModelHealthResponse
from src.services.inference_service import ModelInferenceService, get_inference_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="API Health Check")
def check_api_health():
    """Verify that the FastAPI service is active and responsive."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/health/models", response_model=ModelHealthResponse, summary="Model Artifacts Health Check")
def check_models_health(service: ModelInferenceService = Depends(get_inference_service)):
    """Verify that model checkpoints and preprocessing configurations are loaded into memory."""
    status_data = service.get_health_status()
    return ModelHealthResponse(**status_data)
