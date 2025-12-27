from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, constr

from src.models import PromptPriority


PromptPriorityLiteral = Literal[
    PromptPriority.HIGH.value,
    PromptPriority.NORMAL.value,
    PromptPriority.LOW.value,
]


class ProcessRequest(BaseModel):
    user_id: constr(min_length=1)
    prompt_id: constr(min_length=1)
    text: constr(min_length=1)
    priority: PromptPriorityLiteral = PromptPriority.NORMAL.value


class ProcessResponse(BaseModel):
    user_id: str
    prompt_id: str
    status: Literal["completed", "queued", "failed", "processing"]
    cached: bool
    response: Optional[str] = None
    processing_time_ms: Optional[int] = None
    retry_count: int = Field(0, ge=0)
    error: Optional[str] = None


class HealthComponentStatus(BaseModel):
    database: str
    worker: str
    cache: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"]
    timestamp: datetime
    components: HealthComponentStatus

