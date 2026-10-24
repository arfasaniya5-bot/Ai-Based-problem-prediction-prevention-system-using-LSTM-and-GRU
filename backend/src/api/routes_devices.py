"""Device registry routes for DeviceGuard AI."""

import math
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import Device
from src.schemas.device import DeviceResponse, PaginatedDevicesResponse

router = APIRouter(prefix="/api", tags=["Devices"])


@router.get(
    "/devices",
    response_model=PaginatedDevicesResponse,
    status_code=http_status.HTTP_200_OK,
    summary="List Registered Devices",
)
def list_devices(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of devices per page"),
    search: Optional[str] = Query(None, max_length=100, description="Filter by serial number substring"),
    status: Optional[str] = Query(None, max_length=50, description="Filter by status: HEALTHY, WARNING, CRITICAL, FAILED"),
    db: Session = Depends(get_db),
):
    """Retrieve paginated storage drive inventory from SQLite.

    Supports substring search by serial number and filtering by operational health status.
    """
    query = db.query(Device)

    VALID_DEVICE_STATUSES = {"HEALTHY", "WARNING", "CRITICAL", "FAILED", "ALL"}

    if search and search.strip():
        query = query.filter(Device.serial_number.ilike(f"%{search.strip()}%"))

    if status and status.strip():
        normalized_status = status.strip().upper()
        if normalized_status not in VALID_DEVICE_STATUSES:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported status filter: '{status}'. Valid options are: HEALTHY, WARNING, CRITICAL, FAILED.",
            )
        if normalized_status != "ALL":
            query = query.filter(Device.status == normalized_status)

    total = query.count()
    total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

    devices = (
        query.order_by(Device.last_seen.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [DeviceResponse.model_validate(d) for d in devices]

    return PaginatedDevicesResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
