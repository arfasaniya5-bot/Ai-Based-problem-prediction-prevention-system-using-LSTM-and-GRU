"""FastAPI application entrypoint for DeviceGuard AI."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes_health import router as health_router
from src.api.routes_prediction import router as prediction_router
from src.api.routes_devices import router as devices_router
from src.api.routes_alerts import router as alerts_router
from src.config import settings
from src.database.connection import init_db
from src.services.inference_service import get_inference_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deviceguard_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info("Initializing DeviceGuard AI Backend...")
    # Initialize database tables
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Warm up inference service & model loading
    try:
        service = get_inference_service()
        health = service.get_health_status()
        logger.info(f"Model Inference Service initialized (Status: {health['status']}).")
    except Exception as e:
        logger.error(f"Failed to load model service: {e}")

    yield

    logger.info("Shutting down DeviceGuard AI Backend...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "AI-Based Problem Prediction and Prevention System Using LSTM and GRU. "
        "Provides REST API endpoints for storage drive failure early warning."
    ),
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(devices_router)
app.include_router(alerts_router)


@app.get("/", summary="Root Service Information")
def root_info():
    """General service information and API discovery."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "online",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "disclaimer": "Academic research prototype; not certified for production deployment.",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler avoiding sensitive internal stack exposure."""
    logger.error(f"Unhandled error processing {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred processing your request."},
    )
