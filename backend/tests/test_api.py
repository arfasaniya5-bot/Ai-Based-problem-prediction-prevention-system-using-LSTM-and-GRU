"""Comprehensive test suite for DeviceGuard AI Backend API, Models, and Database."""

from datetime import datetime, timezone
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure sys.path includes backend directory
import sys
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from src.main import app
from src.database.models import Device, PredictionRecord, AlertRecord, TelemetryReading
from tests.conftest import TestingSessionLocal


@pytest.fixture
def sample_valid_readings():
    """Valid 3-day telemetry readings matching training feature schema."""
    return [
        {
            "smart_1_raw": 114175000.0,
            "smart_4_raw": 4.0,
            "smart_5_raw": 0.0,
            "smart_7_raw": 32000.0,
            "smart_9_raw": 8760.0,
            "smart_12_raw": 4.0,
            "smart_187_raw": 0.0,
            "smart_188_raw": 0.0,
            "smart_194_raw": 24.0,
            "smart_197_raw": 0.0,
            "smart_198_raw": 0.0,
        },
        {
            "smart_1_raw": 114210000.0,
            "smart_4_raw": 4.0,
            "smart_5_raw": 0.0,
            "smart_7_raw": 32100.0,
            "smart_9_raw": 8784.0,
            "smart_12_raw": 4.0,
            "smart_187_raw": 0.0,
            "smart_188_raw": 0.0,
            "smart_194_raw": 25.0,
            "smart_197_raw": 0.0,
            "smart_198_raw": 0.0,
        },
        {
            "smart_1_raw": 114250000.0,
            "smart_4_raw": 4.0,
            "smart_5_raw": 0.0,
            "smart_7_raw": 32200.0,
            "smart_9_raw": 8808.0,
            "smart_12_raw": 4.0,
            "smart_187_raw": 0.0,
            "smart_188_raw": 0.0,
            "smart_194_raw": 25.0,
            "smart_197_raw": 0.0,
            "smart_198_raw": 0.0,
        },
    ]


# -----------------------------------------------------------------------------
# 1. Health & Discovery Endpoints
# -----------------------------------------------------------------------------
def test_root_endpoint(client):
    """GET / returns general service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "DeviceGuard AI Backend"
    assert data["status"] == "online"
    assert "version" in data
    assert "docs_url" in data


def test_api_health_endpoint(client):
    """GET /health returns active health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_models_health_endpoint(client):
    """GET /health/models confirms model artifacts are loaded in memory."""
    response = client.get("/health/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loaded"
    assert data["models_loaded"]["lstm"] is True
    assert data["feature_count"] == 11
    assert data["sequence_length"] == 3
    assert data["horizon_days"] == 2
    assert data["decision_threshold"] == 0.50


# -----------------------------------------------------------------------------
# 2. Prediction Endpoints & Validation
# -----------------------------------------------------------------------------
def test_valid_prediction(client, sample_valid_readings):
    """POST /api/predict returns valid failure risk prediction."""
    payload = {
        "serial_number": "TEST_DRIVE_001",
        "model": "ST4000DM000",
        "readings": sample_valid_readings,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["serial_number"] == "TEST_DRIVE_001"
    assert data["model_name"] == "LSTM Neural Network"
    assert data["prediction_class"] in [0, 1]
    assert isinstance(data["is_failure_predicted"], bool)
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] in ["LOW", "MODERATE", "CRITICAL"]
    assert data["threshold"] == 0.50
    assert data["horizon_days"] == 2
    assert data["sequence_length_days"] == 3
    assert len(data["recommendation"]) > 10


def test_prediction_with_null_features(client, sample_valid_readings):
    """POST /api/predict imputes missing features using training medians."""
    # Set several features to null
    sample_valid_readings[0]["smart_187_raw"] = None
    sample_valid_readings[1]["smart_194_raw"] = None
    sample_valid_readings[2]["smart_197_raw"] = None

    payload = {
        "serial_number": "TEST_DRIVE_NULLS",
        "readings": sample_valid_readings,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_class"] in [0, 1]
    assert 0.0 <= data["risk_score"] <= 1.0


def test_invalid_sequence_length_too_short(client, sample_valid_readings):
    """POST /api/predict rejects sequences with fewer than 3 timesteps."""
    payload = {
        "serial_number": "SHORT_SEQ",
        "readings": sample_valid_readings[:2],  # Only 2 days
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "exactly 3 consecutive" in response.text


def test_invalid_sequence_length_too_long(client, sample_valid_readings):
    """POST /api/predict rejects sequences with more than 3 timesteps."""
    payload = {
        "serial_number": "LONG_SEQ",
        "readings": sample_valid_readings + [sample_valid_readings[0]],  # 4 days
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "exactly 3 consecutive" in response.text


def test_invalid_non_numeric_feature(client, sample_valid_readings):
    """POST /api/predict rejects non-numeric string values."""
    sample_valid_readings[0]["smart_5_raw"] = "corrupted_non_numeric"
    payload = {
        "readings": sample_valid_readings,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# 3. Database Persistence Tests
# -----------------------------------------------------------------------------
def test_database_persistence_and_history(client, sample_valid_readings):
    """Verify that predictions are persisted and queryable via /api/predictions."""
    unique_serial = f"AUDIT_DRIVE_{int(datetime.now(timezone.utc).timestamp())}"
    payload = {
        "serial_number": unique_serial,
        "model": "ST4000DM000",
        "readings": sample_valid_readings,
    }
    pred_res = client.post("/api/predict", json=payload)
    assert pred_res.status_code == 200

    # Query history
    hist_res = client.get("/api/predictions?limit=10")
    assert hist_res.status_code == 200
    records = hist_res.json()
    assert len(records) > 0
    assert any(r["model_name"] == "LSTM Neural Network" for r in records)

    # Verify directly via SQLAlchemy session in isolated test DB
    db = TestingSessionLocal()
    try:
        device = db.query(Device).filter(Device.serial_number == unique_serial).first()
        assert device is not None
        assert device.model == "ST4000DM000"
        assert device.status in ["HEALTHY", "WARNING", "CRITICAL"]

        # Verify telemetry readings are persisted
        readings = db.query(TelemetryReading).filter(TelemetryReading.device_id == device.id).all()
        assert len(readings) == 3
    finally:
        db.close()


def test_database_error_handling_predict(client, sample_valid_readings, monkeypatch):
    """Simulate a database failure during prediction persistence and confirm HTTP 500 error."""
    from sqlalchemy.orm import Session

    def mock_commit(self):
        raise RuntimeError("Simulated SQLite write/lock failure")

    monkeypatch.setattr(Session, "commit", mock_commit)

    payload = {
        "serial_number": "FAIL_DRIVE_TEST",
        "model": "ST4000DM000",
        "readings": sample_valid_readings,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 500
    assert "database persistence failed" in res.json()["detail"].lower()


def test_predict_oversized_serial_number(client, sample_valid_readings):
    """POST /api/predict rejects serial numbers exceeding 64 characters with HTTP 422."""
    payload = {
        "serial_number": "A" * 65,
        "model": "ST4000DM000",
        "readings": sample_valid_readings,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
    assert "serial_number" in res.text

