from datetime import datetime, timezone
import logging
import math
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

logger = logging.getLogger(__name__)

from src.database.connection import get_db
from src.database.models import AlertRecord, Device
from src.schemas.alert import AlertAcknowledgeResponse, AlertResponse, PaginatedAlertsResponse

router = APIRouter(prefix="/api", tags=["Alerts"])


@router.get(
    "/alerts",
    response_model=PaginatedAlertsResponse,
    status_code=status.HTTP_200_OK,
    summary="List Hardware Failure Alerts",
)
def list_alerts(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Alerts per page"),
    severity: Optional[str] = Query(None, max_length=50, description="Filter by severity: INFO, WARNING, CRITICAL"),
    acknowledged: Optional[bool] = Query(None, description="Filter by operator acknowledgement status"),
    db: Session = Depends(get_db),
):
    """Retrieve failure prevention alerts from SQLite.

    Supports filtering by severity level and operator acknowledgement status.
    """
    query = db.query(AlertRecord).options(joinedload(AlertRecord.device))

    VALID_SEVERITIES = {"INFO", "WARNING", "CRITICAL", "ALL"}

    if severity and severity.strip():
        normalized_severity = severity.strip().upper()
        if normalized_severity not in VALID_SEVERITIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported severity filter: '{severity}'. Valid options are: INFO, WARNING, CRITICAL.",
            )
        if normalized_severity != "ALL":
            query = query.filter(AlertRecord.severity == normalized_severity)

    if acknowledged is not None:
        query = query.filter(AlertRecord.acknowledged == acknowledged)

    total = query.count()
    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

    alerts = (
        query.order_by(AlertRecord.timestamp.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [
        AlertResponse(
            id=a.id,
            prediction_id=a.prediction_id,
            device_id=a.device_id,
            device_serial=a.device.serial_number if a.device else None,
            device_model=a.device.model if a.device else None,
            timestamp=a.timestamp,
            severity=a.severity,
            recommendation=a.recommendation,
            acknowledged=a.acknowledged,
            acknowledged_at=a.acknowledged_at,
        )
        for a in alerts
    ]

    return PaginatedAlertsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.patch(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertAcknowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Acknowledge Alert Record",
)
def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
):
    """Mark a proactive failure alert as acknowledged by an operator.

    Idempotent operation: repeated calls safely confirm the persisted status.
    """
    alert = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert record with ID {alert_id} not found in database.",
        )

    if not alert.acknowledged:
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now(timezone.utc)
    elif alert.acknowledged_at is None:
        alert.acknowledged_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(alert)
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist alert acknowledgement for ID {alert_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database persistence failed while acknowledging alert.",
        )

    return AlertAcknowledgeResponse(
        id=alert.id,
        acknowledged=alert.acknowledged,
        acknowledged_at=alert.acknowledged_at,
        message=f"Alert #{alert.id} successfully acknowledged.",
    )
