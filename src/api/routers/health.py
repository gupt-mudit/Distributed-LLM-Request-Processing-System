from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
import redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.dependencies import get_db_session
from src.worker import celery_app

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check(session: Session = Depends(get_db_session)) -> dict[str, object]:
    postgres_status = "connected"
    redis_status = "connected"
    worker_status = "unknown"

    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - connectivity failure path
        postgres_status = f"error: {exc}"

    try:
        redis_client = redis.Redis.from_url("redis://redis:6379/0")
        redis_client.ping()
    except Exception as exc:  # pragma: no cover - redis failure
        redis_status = f"error: {exc}"

    try:
        celery_app.control.ping(timeout=1.0)
        worker_status = "running"
    except Exception as exc:  # pragma: no cover - Celery control failure
        worker_status = f"error: {exc}"

    status = (
        postgres_status == "connected"
        and worker_status == "running"
        and redis_status == "connected"
    )
    if not status:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "database": postgres_status,
                    "worker": worker_status,
                    "cache": redis_status,
                },
            },
        )

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": postgres_status,
            "worker": worker_status,
            "cache": redis_status,
        },
    }

