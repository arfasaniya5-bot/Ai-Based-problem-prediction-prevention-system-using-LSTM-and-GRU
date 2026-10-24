"""Pydantic schemas for failure prevention alerts and acknowledgement."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AlertResponse(BaseModel):
    """Failure early-warning alert with joined device details."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique database alert record ID")
    prediction_id: int = Field(..., description="Foreign key to triggering prediction record")
    device_id: Optional[int] = Field(None, description="Associated storage drive ID")
    device_serial: Optional[str] = Field(None, description="Joined device serial number")
    device_model: Optional[str] = Field(None, description="Joined hardware model string")
    timestamp: datetime = Field(..., description="Timestamp when alert was triggered")
    severity: str = Field(..., description="Severity category: INFO, WARNING, CRITICAL")
    recommendation: str = Field(..., description="Actionable preventative maintenance recommendation")
    acknowledged: bool = Field(..., description="Whether an operator has acknowledged the alert")
    acknowledged_at: Optional[datetime] = Field(None, description="UTC timestamp of operator acknowledgement")


class PaginatedAlertsResponse(BaseModel):
    """Paginated list of failure prevention alerts."""

    items: List[AlertResponse] = Field(..., description="List of alerts on current page")
    total: int = Field(..., description="Total matching alert records")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages available")


class AlertAcknowledgeResponse(BaseModel):
    """Result of an alert acknowledgement mutation."""

    id: int = Field(..., description="Alert ID")
    acknowledged: bool = Field(True, description="Updated acknowledgement status")
    acknowledged_at: Optional[datetime] = Field(..., description="Timestamp of acknowledgement")
    message: str = Field(..., description="Status summary message")
