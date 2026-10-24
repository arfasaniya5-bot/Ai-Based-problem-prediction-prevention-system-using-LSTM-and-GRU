"""Automated test suite for Device Registry and Alert Lifecycle APIs (Part 7)."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from src.main import app
from src.database.models import Device, AlertRecord, PredictionRecord
from tests.conftest import TestingSessionLocal


@pytest.fixture(scope="module")
def seeded_db():
    """Ensures test devices and alert records exist in the isolated test database."""
    db = TestingSessionLocal()
    try:
        # Create test device
        test_serial = f"TEST_REGISTRY_{int(datetime.now(timezone.utc).timestamp())}"
        device = Device(
            serial_number=test_serial,
            model="ST4000DM000",
            status="HEALTHY",
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
        )
        db.add(device)
        db.flush()

        # Create prediction and alert
        pred = PredictionRecord(
            device_id=device.id,
            timestamp=datetime.now(timezone.utc),
            model_name="LSTM Neural Network",
            model_version="1.0.0",
            prediction_class=1,
            risk_score=0.885,
            risk_level="CRITICAL",
            threshold=0.50,
            horizon_days=2,
            input_sequence_length=3,
            status="SUCCESS",
        )
        db.add(pred)
        db.flush()

        alert = AlertRecord(
            prediction_id=pred.id,
            device_id=device.id,
            timestamp=datetime.now(timezone.utc),
            severity="CRITICAL",
            recommendation="Test recommendation: Replace drive.",
            acknowledged=False,
        )
        db.add(alert)
        db.commit()

        yield {
            "device_id": device.id,
            "serial_number": test_serial,
            "prediction_id": pred.id,
            "alert_id": alert.id,
        }
    finally:
        db.close()


# -----------------------------------------------------------------------------
# 1. Device Registry Tests
# -----------------------------------------------------------------------------
def test_list_devices_pagination(client, seeded_db):
    """GET /api/devices returns a paginated structure with metadata."""
    res = client.get("/api/devices?page=1&page_size=5")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "total_pages" in data
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert data["total"] >= 1
    assert len(data["items"]) <= 5

    # Check device schema
    item = data["items"][0]
    assert "id" in item
    assert "serial_number" in item
    assert "status" in item
    assert "first_seen" in item
    assert "last_seen" in item


def test_list_devices_search(client, seeded_db):
    """GET /api/devices filters by serial number substring."""
    target_serial = seeded_db["serial_number"]
    # Exact or substring search
    res = client.get(f"/api/devices?search={target_serial[:12]}")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1
    assert any(d["serial_number"] == target_serial for d in data["items"])


def test_list_devices_status_filter(client, seeded_db):
    """GET /api/devices filters by status."""
    res = client.get("/api/devices?status=HEALTHY")
    assert res.status_code == 200
    data = res.json()
    assert all(d["status"] == "HEALTHY" for d in data["items"])


def test_list_devices_empty_search(client):
    """GET /api/devices returns empty list when no serial number matches."""
    res = client.get("/api/devices?search=NONEXISTENT_SERIAL_99999999")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0
    assert data["total_pages"] == 1


# -----------------------------------------------------------------------------
# 2. Alert Listing Tests
# -----------------------------------------------------------------------------
def test_list_alerts_structure(client, seeded_db):
    """GET /api/alerts returns paginated alerts with joined device info."""
    res = client.get("/api/alerts?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1

    # Check alert item schema
    target_alert = next((a for a in data["items"] if a["id"] == seeded_db["alert_id"]), None)
    assert target_alert is not None
    assert target_alert["device_id"] == seeded_db["device_id"]
    assert target_alert["device_serial"] == seeded_db["serial_number"]
    assert target_alert["severity"] == "CRITICAL"
    assert "recommendation" in target_alert


def test_list_alerts_severity_filter(client, seeded_db):
    """GET /api/alerts filters by severity."""
    res = client.get("/api/alerts?severity=CRITICAL")
    assert res.status_code == 200
    data = res.json()
    assert all(a["severity"] == "CRITICAL" for a in data["items"])


def test_list_alerts_acknowledged_filter(client, seeded_db):
    """GET /api/alerts filters by acknowledgement status."""
    res_unack = client.get("/api/alerts?acknowledged=false")
    assert res_unack.status_code == 200
    data_unack = res_unack.json()
    assert all(a["acknowledged"] is False for a in data_unack["items"])


# -----------------------------------------------------------------------------
# 3. Alert Acknowledgement Lifecycle Tests
# -----------------------------------------------------------------------------
def test_acknowledge_alert_success(client, seeded_db):
    """PATCH /api/alerts/{id}/acknowledge marks an alert as acknowledged and sets timestamp."""
    alert_id = seeded_db["alert_id"]
    res = client.patch(f"/api/alerts/{alert_id}/acknowledge")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == alert_id
    assert data["acknowledged"] is True
    assert data["acknowledged_at"] is not None

    # Verify query reflects acknowledged status
    query_res = client.get(f"/api/alerts?page=1&page_size=50")
    assert query_res.status_code == 200
    updated_alert = next(a for a in query_res.json()["items"] if a["id"] == alert_id)
    assert updated_alert["acknowledged"] is True
    assert updated_alert["acknowledged_at"] is not None


def test_acknowledge_alert_idempotency(client, seeded_db):
    """Repeated calls to PATCH /api/alerts/{id}/acknowledge safely return 200."""
    alert_id = seeded_db["alert_id"]
    res1 = client.patch(f"/api/alerts/{alert_id}/acknowledge")
    assert res1.status_code == 200

    res2 = client.patch(f"/api/alerts/{alert_id}/acknowledge")
    assert res2.status_code == 200
    assert res2.json()["acknowledged"] is True
    assert res2.json()["acknowledged_at"] is not None


def test_acknowledge_alert_not_found(client):
    """PATCH /api/alerts/{id}/acknowledge returns 404 for nonexistent alert."""
    res = client.patch("/api/alerts/999999/acknowledge")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 4. Error Handling and Edge Case Regression Tests (Part 8)
# -----------------------------------------------------------------------------
def test_unsupported_device_status_filter(client):
    """GET /api/devices returns HTTP 400 for unsupported status values."""
    res = client.get("/api/devices?status=INVALID_STATUS_FOOBAR")
    assert res.status_code == 400
    assert "unsupported status" in res.json()["detail"].lower()


def test_unsupported_alert_severity_filter(client):
    """GET /api/alerts returns HTTP 400 for unsupported severity values."""
    res = client.get("/api/alerts?severity=INVALID_SEVERITY_LEVEL")
    assert res.status_code == 400
    assert "unsupported severity" in res.json()["detail"].lower()


def test_invalid_device_pagination_page_zero(client):
    """GET /api/devices returns HTTP 422 for page=0 (violates ge=1)."""
    res = client.get("/api/devices?page=0")
    assert res.status_code == 422


def test_invalid_device_pagination_page_size_exceeded(client):
    """GET /api/devices returns HTTP 422 for page_size=500 (violates le=100)."""
    res = client.get("/api/devices?page_size=500")
    assert res.status_code == 422


def test_duplicate_alert_prevention_on_repeated_predict(client):
    """Repeated predictions for the same degraded drive update the existing unacknowledged alert rather than creating duplicate rows."""
    degraded_serial = f"DEDUP_TEST_{int(datetime.now(timezone.utc).timestamp())}"
    readings = [
        {"smart_1_raw": 8941020, "smart_4_raw": 14, "smart_5_raw": 16, "smart_7_raw": 890120, "smart_9_raw": 28400, "smart_12_raw": 14, "smart_187_raw": 12, "smart_188_raw": 4, "smart_194_raw": 34, "smart_197_raw": 8, "smart_198_raw": 8},
        {"smart_1_raw": 14502100, "smart_4_raw": 14, "smart_5_raw": 64, "smart_7_raw": 1450300, "smart_9_raw": 28424, "smart_12_raw": 14, "smart_187_raw": 45, "smart_188_raw": 9, "smart_194_raw": 38, "smart_197_raw": 24, "smart_198_raw": 24},
        {"smart_1_raw": 28941500, "smart_4_raw": 14, "smart_5_raw": 180, "smart_7_raw": 2894000, "smart_9_raw": 28448, "smart_12_raw": 14, "smart_187_raw": 128, "smart_188_raw": 22, "smart_194_raw": 42, "smart_197_raw": 60, "smart_198_raw": 60},
    ]

    # First prediction
    payload = {"serial_number": degraded_serial, "model": "ST4000DM000", "readings": readings}
    res1 = client.post("/api/predict", json=payload)
    assert res1.status_code == 200

    # Query alerts for this device
    alerts_res1 = client.get("/api/alerts?page=1&page_size=50")
    assert alerts_res1.status_code == 200
    device_alerts1 = [a for a in alerts_res1.json()["items"] if a["device_serial"] == degraded_serial and not a["acknowledged"]]
    assert len(device_alerts1) == 1
    alert_id_first = device_alerts1[0]["id"]

    # Second prediction for the same device
    res2 = client.post("/api/predict", json=payload)
    assert res2.status_code == 200

    # Query alerts again - must still be exactly 1 unacknowledged alert (updated, not duplicated)
    alerts_res2 = client.get("/api/alerts?page=1&page_size=50")
    assert alerts_res2.status_code == 200
    device_alerts2 = [a for a in alerts_res2.json()["items"] if a["device_serial"] == degraded_serial and not a["acknowledged"]]
    assert len(device_alerts2) == 1
    assert device_alerts2[0]["id"] == alert_id_first


def test_device_status_consistency_warning(client):
    """Device status is set to WARNING when telemetry exhibits moderate wear risk."""
    mod_serial = f"MOD_WEAR_TEST_{int(datetime.now(timezone.utc).timestamp())}"
    readings = [
        {"smart_1_raw": 8941020, "smart_4_raw": 14, "smart_5_raw": 16, "smart_7_raw": 890120, "smart_9_raw": 28400, "smart_12_raw": 14, "smart_187_raw": 12, "smart_188_raw": 4, "smart_194_raw": 34, "smart_197_raw": 8, "smart_198_raw": 8},
        {"smart_1_raw": 14502100, "smart_4_raw": 14, "smart_5_raw": 64, "smart_7_raw": 1450300, "smart_9_raw": 28424, "smart_12_raw": 14, "smart_187_raw": 45, "smart_188_raw": 9, "smart_194_raw": 38, "smart_197_raw": 24, "smart_198_raw": 24},
        {"smart_1_raw": 28941500, "smart_4_raw": 14, "smart_5_raw": 180, "smart_7_raw": 2894000, "smart_9_raw": 28448, "smart_12_raw": 14, "smart_187_raw": 128, "smart_188_raw": 22, "smart_194_raw": 42, "smart_197_raw": 60, "smart_198_raw": 60},
    ]

    payload = {"serial_number": mod_serial, "model": "ST4000DM000", "readings": readings}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200

    # Query device from registry
    dev_res = client.get(f"/api/devices?search={mod_serial}")
    assert dev_res.status_code == 200
    assert dev_res.json()["total"] == 1
    device_data = dev_res.json()["items"][0]
    assert device_data["status"] == "WARNING"


def test_devices_oversized_search_string(client):
    """GET /api/devices rejects search parameters exceeding 100 characters with HTTP 422."""
    oversized_search = "A" * 101
    res = client.get(f"/api/devices?search={oversized_search}")
    assert res.status_code == 422
    assert "search" in res.text


def test_acknowledge_alert_database_error(client, seeded_db, monkeypatch):
    """PATCH /api/alerts/{id}/acknowledge rolls back and returns HTTP 500 on database failure."""
    from sqlalchemy.orm import Session

    def mock_commit(self):
        raise RuntimeError("Simulated SQLite write/lock failure during acknowledgement")

    monkeypatch.setattr(Session, "commit", mock_commit)

    alert_id = seeded_db["alert_id"]
    res = client.patch(f"/api/alerts/{alert_id}/acknowledge")
    assert res.status_code == 500
    assert "database persistence failed" in res.json()["detail"].lower()

