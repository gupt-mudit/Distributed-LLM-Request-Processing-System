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
from src.services.semantic_cache import CacheHit, SemanticCacheService

router = APIRouter(prefix="/process", tags=["processing"])

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def _build_response_from_existing(
    existing: dict,
    cache_service: SemanticCacheService,
    start_time: float,
) -> ProcessResponse:
    """Build ProcessResponse from existing MongoDB document.
    
    Args:
        existing: MongoDB document.
        cache_service: Semantic cache service.
        start_time: Start time for processing time calculation.
        
    Returns:
        ProcessResponse object.
    """
    cache_entry_id = existing.get("cache_entry_id")
    response_text = cache_service.get_response_text(cache_entry_id) if cache_entry_id else None
    
    return ProcessResponse(
        user_id=existing["user_id"],
        prompt_id=existing["prompt_id"],
        status=existing.get("status", PromptStatus.QUEUED.value),
        cached=existing.get("cached", False),
        response=response_text,
        processing_time_ms=existing.get("processing_time_ms"),
        retry_count=existing.get("retry_count", 0),
        error=existing.get("error_message"),
    )


def _handle_completed_with_cache(
    existing: dict,
    payload: ProcessRequest,
    cache_service: SemanticCacheService,
    start_time: float,
) -> Optional[ProcessResponse]:
    """Handle completed request with cache entry.
    
    Args:
        existing: MongoDB document.
        payload: Process request payload.
        cache_service: Semantic cache service.
        start_time: Start time for processing time calculation.
        
    Returns:
        ProcessResponse if cache hit, None otherwise.
    """
    cache_entry_id = existing.get("cache_entry_id")
    if not cache_entry_id:
        return None
    
    response_text = cache_service.get_response_text(cache_entry_id)
    if response_text:
        cache_service.record_hit(cache_entry_id)
        return ProcessResponse(
            user_id=payload.user_id,
            prompt_id=payload.prompt_id,
            status=PromptStatus.COMPLETED.value,
            cached=True,
            response=response_text,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
            retry_count=existing.get("retry_count", 0),
        )
    return None


def _handle_existing_request(
    existing: dict,
    payload: ProcessRequest,
    cache_service: SemanticCacheService,
    start_time: float,
) -> Optional[ProcessResponse]:
    """Handle existing request based on status.
    
    Args:
        existing: MongoDB document.
        payload: Process request payload.
        cache_service: Semantic cache service.
        start_time: Start time for processing time calculation.
        
    Returns:
        ProcessResponse if handled, None if should continue processing.
    """
    status = existing.get("status")
    
    # Completed with cache - try to return cached response
    if status == PromptStatus.COMPLETED.value and existing.get("cache_entry_id"):
        response = _handle_completed_with_cache(
            existing, payload, cache_service, start_time
        )
        if response:
            return response
        # Cache retrieval failed, continue to semantic cache lookup
    
    # Queued or processing - return current status
    if status in {PromptStatus.QUEUED.value, PromptStatus.PROCESSING.value}:
        return _build_response_from_existing(existing, cache_service, start_time)
    
    # Failed - return failed status
    if status == PromptStatus.FAILED.value:
        return _build_response_from_existing(existing, cache_service, start_time)
    
    return None


def _handle_cache_hit(
    cache_hit: CacheHit,
    payload: ProcessRequest,
    collection: Collection,
    cache_service: SemanticCacheService,
    existing: Optional[dict],
    start_time: float,
) -> ProcessResponse:
    """Handle semantic cache hit.
    
    Args:
        cache_hit: Cache hit result.
        payload: Process request payload.
        collection: MongoDB collection.
        cache_service: Semantic cache service.
        existing: Existing MongoDB document (if any).
        start_time: Start time for processing time calculation.
        
    Returns:
        ProcessResponse with cached result.
    """
    cache_service.record_hit(cache_hit.cache_entry_id)
    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
    
    _persist_prompt_request(
        collection=collection,
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


def _enqueue_new_request(
    payload: ProcessRequest,
    collection: Collection,
    task_client,
    existing: Optional[dict],
) -> None:
    """Create request document and enqueue Celery task.
    
    Args:
        payload: Process request payload.
        collection: MongoDB collection.
        task_client: Task client for enqueuing.
        existing: Existing MongoDB document (if any).
    """
    now = datetime.now(timezone.utc)
    request_doc = {
        "user_id": payload.user_id,
        "prompt_id": payload.prompt_id,
        "prompt_text": payload.text.strip(),
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
    
    collection.update_one(
        {"user_id": payload.user_id, "prompt_id": payload.prompt_id},
        {"$set": request_doc},
        upsert=True,
    )
    
    task_client.enqueue(
        user_id=payload.user_id,
        prompt_id=payload.prompt_id,
        text=payload.text.strip(),
        priority=payload.priority,
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("", response_model=ProcessResponse)
def process_prompt(
    payload: ProcessRequest,
    collection: Collection = Depends(get_db_collection),
    embedding_service=Depends(get_embedding_service),
    cache_service: SemanticCacheService = Depends(get_cache_service),
    task_client=Depends(get_prompt_task_client),
) -> ProcessResponse:
    """Process a prompt request.
    
    Returns immediately with queued status if cache miss,
    or completed status if cache hit.
    """
    start_time = time.perf_counter()
    
    # Find existing request
    existing = collection.find_one(
        {"user_id": payload.user_id, "prompt_id": payload.prompt_id}
    )
    
    # Handle existing requests
    if existing:
        response = _handle_existing_request(
            existing, payload, cache_service, start_time
        )
        if response:
            return response
    
    # New request or cache miss - generate embedding and check cache
    prompt_text = payload.text.strip()
    embedding = embedding_service.embed(prompt_text)
    cache_hit = cache_service.lookup(embedding)
    
    # Handle cache hit
    if cache_hit:
        return _handle_cache_hit(
            cache_hit=cache_hit,
            payload=payload,
            collection=collection,
            cache_service=cache_service,
            existing=existing,
            start_time=start_time,
        )
    
    # Cache miss - enqueue for background processing
    _enqueue_new_request(payload, collection, task_client, existing)
    
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
    cache_service: SemanticCacheService = Depends(get_cache_service),
) -> ProcessResponse:
    """Get the status of a prompt request."""
    request = collection.find_one({"user_id": user_id, "prompt_id": prompt_id})
    
    if request is None:
        raise HTTPException(
            status_code=404,
            detail=f"PromptRequest not found: user_id={user_id}, prompt_id={prompt_id}",
        )
    
    return _build_response_from_existing(request, cache_service, 0.0)


# ============================================================================
# Database Operations
# ============================================================================

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
