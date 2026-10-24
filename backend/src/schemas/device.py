"""Pydantic schemas for device registry and pagination."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DeviceResponse(BaseModel):
    """Storage drive representation returned by registry endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique database identifier")
    serial_number: str = Field(..., description="Hard drive serial number")
    model: Optional[str] = Field(None, description="Hardware model string")
    capacity_bytes: Optional[int] = Field(None, description="Reported capacity in bytes")
    status: str = Field(..., description="HEALTHY, WARNING, CRITICAL, or FAILED")
    first_seen: datetime = Field(..., description="UTC timestamp of first telemetry observation")
    last_seen: datetime = Field(..., description="UTC timestamp of most recent telemetry observation")


class PaginatedDevicesResponse(BaseModel):
    """Paginated list of registered storage drives with metadata."""

    items: List[DeviceResponse] = Field(..., description="List of devices on current page")
    total: int = Field(..., description="Total matching device records")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages available")
