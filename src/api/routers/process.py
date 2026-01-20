from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pymongo.collection import Collection

from src.api.dependencies import (
    get_cache_service,
    get_db_collection,
    get_embedding_service,
    get_prompt_task_client,
)
from src.api.schemas import ProcessRequest, ProcessResponse
from src.models.mongodb_models import PromptPriority, PromptStatus
from src.services.semantic_cache import CacheHit

router = APIRouter(prefix="/process", tags=["processing"])

logger = logging.getLogger(__name__)


@router.post("", response_model=ProcessResponse)
def process_prompt(
    payload: ProcessRequest,
    collection: Collection = Depends(get_db_collection),
    embedding_service=Depends(get_embedding_service),
    cache_service=Depends(get_cache_service),
    task_client=Depends(get_prompt_task_client),
) -> ProcessResponse:
    start_time = time.perf_counter()

    # Find existing request (atomic operation)
    existing = collection.find_one(
        {"user_id": payload.user_id, "prompt_id": payload.prompt_id}
    )

    # If completed and has cache entry, return cached response
    if (
        existing
        and existing.get("status") == PromptStatus.COMPLETED.value
        and existing.get("cache_entry_id")
    ):
        try:
            cache_point = cache_service._qdrant.get(existing["cache_entry_id"])
            if cache_point:
                cache_service.record_hit(existing["cache_entry_id"])
                return ProcessResponse(
                    user_id=payload.user_id,
                    prompt_id=payload.prompt_id,
                    status=PromptStatus.COMPLETED.value,
                    cached=True,
                    response=cache_point["payload"].get("response_text", ""),
                    processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                    retry_count=existing.get("retry_count", 0),
                )
        except Exception as exc:
            logger.warning(
                "Failed to retrieve cache entry, will continue processing",
                extra={
                    "user_id": payload.user_id,
                    "prompt_id": payload.prompt_id,
                    "error": str(exc),
                },
            )
            # Continue to cache lookup below

    # If queued or processing, return current status
    if existing and existing.get("status") in {
        PromptStatus.QUEUED.value,
        PromptStatus.PROCESSING.value,
    }:
        response_text = None
        if existing.get("cache_entry_id"):
            try:
                cache_point = cache_service._qdrant.get(existing["cache_entry_id"])
                response_text = (
                    cache_point["payload"].get("response_text") if cache_point else None
                )
            except Exception as exc:
                logger.warning(
                    "Failed to retrieve cache entry for queued/processing request",
                    extra={
                        "user_id": existing.get("user_id"),
                        "prompt_id": existing.get("prompt_id"),
                        "error": str(exc),
                    },
                )
                response_text = None

        return ProcessResponse(
            user_id=existing["user_id"],
            prompt_id=existing["prompt_id"],
            status=existing["status"],
            cached=existing.get("cached", False),
            response=response_text,
            processing_time_ms=existing.get("processing_time_ms"),
            retry_count=existing.get("retry_count", 0),
            error=existing.get("error_message"),
        )

    # If failed, return failed status
    if existing and existing.get("status") == PromptStatus.FAILED.value:
        return ProcessResponse(
            user_id=existing["user_id"],
            prompt_id=existing["prompt_id"],
            status=PromptStatus.FAILED.value,
            cached=False,
            response=None,
            processing_time_ms=existing.get("processing_time_ms"),
            retry_count=existing.get("retry_count", 0),
            error=existing.get("error_message"),
        )

    # Generate embedding and check cache
    prompt_text = payload.text.strip()
    embedding = embedding_service.embed(prompt_text)
    cache_hit = cache_service.lookup(embedding)

    if cache_hit:
        cache_service.record_hit(cache_hit.cache_entry_id)
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        _persist_prompt_request(
            collection,
            payload=payload,
            status=PromptStatus.COMPLETED,
            cache_entry_id=cache_hit.cache_entry_id,
            cached=True,
            processing_time_ms=processing_time_ms,
            retry_count=existing.get("retry_count", 0) if existing else 0,
        )
        return ProcessResponse(
            user_id=payload.user_id,
            prompt_id=payload.prompt_id,
            status=PromptStatus.COMPLETED.value,
            cached=True,
            response=cache_hit.response_text,
            processing_time_ms=processing_time_ms,
            retry_count=existing.get("retry_count", 0) if existing else 0,
        )

    # Cache miss - create request and enqueue task
    now = datetime.now(timezone.utc)
    request_doc = {
        "user_id": payload.user_id,
        "prompt_id": payload.prompt_id,
        "prompt_text": prompt_text,
        "status": PromptStatus.QUEUED.value,
        "priority": payload.priority,
        "cached": False,
        "retry_count": existing.get("retry_count", 0) if existing else 0,
        "processing_time_ms": None,
        "error_message": None,
        "cache_entry_id": None,
        "created_at": existing.get("created_at", now) if existing else now,
        "updated_at": now,
    }

    # Upsert (atomic operation)
    collection.update_one(
        {"user_id": payload.user_id, "prompt_id": payload.prompt_id},
        {"$set": request_doc},
        upsert=True,
    )

    # Enqueue task for background processing
    task_client.enqueue(
        user_id=payload.user_id,
        prompt_id=payload.prompt_id,
        text=prompt_text,
        priority=payload.priority,
    )

    # Return immediately - client should poll GET /process/{user_id}/{prompt_id} for status
    return ProcessResponse(
        user_id=payload.user_id,
        prompt_id=payload.prompt_id,
        status=PromptStatus.QUEUED.value,
        cached=False,
        response=None,
        processing_time_ms=None,
        retry_count=existing.get("retry_count", 0) if existing else 0,
        error=None,
    )


@router.get("/{user_id}/{prompt_id}", response_model=ProcessResponse)
def get_prompt_status(
    user_id: str,
    prompt_id: str,
    collection: Collection = Depends(get_db_collection),
    cache_service=Depends(get_cache_service),
) -> ProcessResponse:
    request = collection.find_one({"user_id": user_id, "prompt_id": prompt_id})

    if request is None:
        raise HTTPException(
            status_code=404,
            detail="PromptRequest not found.",
        )

    response_text = None
    if request.get("cache_entry_id"):
        try:
            cache_point = cache_service._qdrant.get(request["cache_entry_id"])
            response_text = (
                cache_point["payload"].get("response_text") if cache_point else None
            )
        except Exception as exc:
            logger.warning(
                "Failed to retrieve cache entry",
                extra={
                    "user_id": user_id,
                    "prompt_id": prompt_id,
                    "error": str(exc),
                },
            )
            response_text = None

    return ProcessResponse(
        user_id=request["user_id"],
        prompt_id=request["prompt_id"],
        status=request.get("status", PromptStatus.QUEUED.value),
        cached=request.get("cached", False),
        response=response_text,
        processing_time_ms=request.get("processing_time_ms"),
        retry_count=request.get("retry_count", 0),
        error=request.get("error_message"),
    )


def _persist_prompt_request(
    collection: Collection,
    *,
    payload: ProcessRequest,
    status: PromptStatus,
    cache_entry_id: Optional[str],
    cached: bool,
    processing_time_ms: Optional[int],
    retry_count: int,
) -> None:
    """Update or create a prompt request document."""
    now = datetime.now(timezone.utc)
    update_doc = {
        "status": status.value,
        "cache_entry_id": cache_entry_id,
        "cached": cached,
        "processing_time_ms": processing_time_ms,
        "retry_count": retry_count,
        "prompt_text": payload.text,
        "updated_at": now,
    }

    collection.update_one(
        {"user_id": payload.user_id, "prompt_id": payload.prompt_id},
        {"$set": update_doc},
        upsert=True,
    )
