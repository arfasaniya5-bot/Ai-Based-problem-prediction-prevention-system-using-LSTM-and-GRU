"""Prediction and inference routes for DeviceGuard AI."""

from datetime import datetime, timezone
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from src.database.connection import get_db
from src.database.models import AlertRecord, Device, PredictionRecord, TelemetryReading
from src.schemas.prediction import (
    PredictionRecordResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.services.inference_service import ModelInferenceService, get_inference_service

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Failure Early-Warning Prediction",
)
def predict_device_failure(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    inference_service: ModelInferenceService = Depends(get_inference_service),
):
    """Predict hard-drive failure within a 2-day horizon using 3 consecutive days of SMART telemetry.

    Validates input features, applies training-fitted median imputation and RobustScaling,
    executes the LSTM neural network forward pass, and persists the prediction log.
    """
    try:
        response = inference_service.predict(
            readings=request.readings,
            serial_number=request.serial_number,
            model_name=request.model,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed processing telemetry.",
        )

    # Database persistence
    try:
        device_id = None
        if request.serial_number:
            device = db.query(Device).filter(Device.serial_number == request.serial_number).first()
            if not device:
                device_status = "CRITICAL" if response.is_failure_predicted else ("WARNING" if response.risk_level == "MODERATE" else "HEALTHY")
                device = Device(
                    serial_number=request.serial_number,
                    model=request.model,
                    status=device_status,
                    first_seen=datetime.now(timezone.utc),
                    last_seen=datetime.now(timezone.utc),
                )
                db.add(device)
                db.flush()
            else:
                device.last_seen = datetime.now(timezone.utc)
                if request.model and not device.model:
                    device.model = request.model
                if response.is_failure_predicted:
                    device.status = "CRITICAL"
                elif response.risk_level == "MODERATE":
                    device.status = "WARNING"
                else:
                    device.status = "HEALTHY"
            device_id = device.id

        # Persist prediction audit record
        pred_record = PredictionRecord(
            device_id=device_id,
            timestamp=response.timestamp,
            model_name=response.model_name,
            model_version=response.model_version,
            prediction_class=response.prediction_class,
            risk_score=response.risk_score,
            risk_level=response.risk_level,
            threshold=response.threshold,
            horizon_days=response.horizon_days,
            input_sequence_length=response.sequence_length_days,
            status="SUCCESS",
        )
        db.add(pred_record)
        db.flush()

        # Persist individual daily telemetry readings if device identified
        if device_id and request.readings:
            for r in request.readings:
                telemetry = TelemetryReading(
                    device_id=device_id,
                    timestamp=response.timestamp,
                    smart_1_raw=r.smart_1_raw,
                    smart_4_raw=r.smart_4_raw,
                    smart_5_raw=r.smart_5_raw,
                    smart_7_raw=r.smart_7_raw,
                    smart_9_raw=r.smart_9_raw,
                    smart_12_raw=r.smart_12_raw,
                    smart_187_raw=r.smart_187_raw,
                    smart_188_raw=r.smart_188_raw,
                    smart_194_raw=r.smart_194_raw,
                    smart_197_raw=r.smart_197_raw,
                    smart_198_raw=r.smart_198_raw,
                )
                db.add(telemetry)

        # Persist failure early-warning alert (preventing unintended duplicate unacknowledged alerts)
        if response.is_failure_predicted or response.risk_level == "MODERATE":
            severity = "CRITICAL" if response.is_failure_predicted else "WARNING"
            existing_unack_alert = (
                db.query(AlertRecord)
                .filter(
                    AlertRecord.device_id == device_id,
                    AlertRecord.severity == severity,
                    AlertRecord.acknowledged == False,
                )
                .first()
            )
            if existing_unack_alert:
                existing_unack_alert.prediction_id = pred_record.id
                existing_unack_alert.timestamp = response.timestamp
                existing_unack_alert.recommendation = response.recommendation
            else:
                alert = AlertRecord(
                    prediction_id=pred_record.id,
                    device_id=device_id,
                    timestamp=response.timestamp,
                    severity=severity,
                    recommendation=response.recommendation,
                    acknowledged=False,
                )
                db.add(alert)

        db.commit()
    except Exception as db_err:
        db.rollback()
        logger.error(f"Database persistence failed: {db_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence failed while recording prediction.",
        )

    return response


@router.get(
    "/predictions",
    response_model=List[PredictionRecordResponse],
    summary="List Prediction History",
)
def list_predictions(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieve historical prediction records stored in the database."""
    records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
    return records
