from __future__ import annotations

import os
from datetime import datetime, timezone

import redis
from fastapi import APIRouter, HTTPException
from pymongo import MongoClient

from src.models.mongodb import get_mongodb_client
from src.worker import celery_app

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> dict[str, object]:
    mongodb_status = "connected"
    redis_status = "connected"
    worker_status = "unknown"
    qdrant_status = "unknown"

    # Check MongoDB
    try:
        client = get_mongodb_client()
        client.admin.command("ping")
    except Exception as exc:  # pragma: no cover - connectivity failure path
        mongodb_status = f"error: {exc}"

    # Check Redis
    try:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_client = redis.Redis.from_url(redis_url)
        redis_client.ping()
    except Exception as exc:  # pragma: no cover - redis failure
        redis_status = f"error: {exc}"

    # Check Celery workers
    try:
        celery_app.control.ping(timeout=1.0)
        worker_status = "running"
    except Exception as exc:  # pragma: no cover - Celery control failure
        worker_status = f"error: {exc}"

    # Check Qdrant
    try:
        from src.api.dependencies import get_cache_service

        cache_service = get_cache_service()
        cache_service._qdrant._client.get_collections()
        qdrant_status = "connected"
    except Exception as exc:  # pragma: no cover - Qdrant failure
        qdrant_status = f"error: {exc}"

    status = (
        mongodb_status == "connected"
        and worker_status == "running"
        and redis_status == "connected"
        and qdrant_status == "connected"
    )
    if not status:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "database": mongodb_status,
                    "worker": worker_status,
                    "cache": redis_status,
                    "vector_db": qdrant_status,
                },
            },
        )

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": mongodb_status,
            "worker": worker_status,
            "cache": redis_status,
            "vector_db": qdrant_status,
        },
    }

