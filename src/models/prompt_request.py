from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    DateTime,
    Enum as PgEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


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


class PromptRequest(Base):
    __tablename__ = "prompt_requests"
    __table_args__ = (
        UniqueConstraint("user_id", "prompt_id", name="uq_prompt_requests_user_prompt"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_id: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[PromptStatus] = mapped_column(
        PgEnum(
            PromptStatus,
            name="prompt_status",
            values_callable=lambda enum: [member.value for member in enum],
        ),
        default=PromptStatus.RECEIVED,
        nullable=False,
    )
    priority: Mapped[PromptPriority] = mapped_column(
        PgEnum(
            PromptPriority,
            name="prompt_priority",
            values_callable=lambda enum: [member.value for member in enum],
        ),
        default=PromptPriority.NORMAL,
        nullable=False,
    )
    cached: Mapped[bool] = mapped_column(default=False, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(String(255))
    cache_entry_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("prompt_cache_entries.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __mapper_args__ = {
        "eager_defaults": True,
    }


