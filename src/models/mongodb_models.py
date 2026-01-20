from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PromptStatus(str, Enum):
    RECEIVED = "received"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PromptPriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class PromptRequestDoc(BaseModel):
    """MongoDB document model for PromptRequest."""

    user_id: str
    prompt_id: str
    prompt_text: str
    status: str = Field(default=PromptStatus.RECEIVED.value)
    priority: str = Field(default=PromptPriority.NORMAL.value)
    cached: bool = False
    retry_count: int = 0
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    cache_entry_id: Optional[str] = None  # Qdrant point ID (string UUID)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "prompt_id": "prompt456",
                "prompt_text": "Explain quantum computing",
                "status": "queued",
                "priority": "normal",
                "cached": False,
                "retry_count": 0,
                "processing_time_ms": None,
                "error_message": None,
                "cache_entry_id": None,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        }

