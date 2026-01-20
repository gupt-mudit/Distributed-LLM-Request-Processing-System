from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict

from celery import Task
from pymongo.errors import DuplicateKeyError

from src.models.mongodb import get_prompt_requests_collection
from src.models.mongodb_models import PromptPriority, PromptStatus
from src.services import (
    EmbeddingService,
    MockLLM,
    ProviderError,
    RateLimitExceeded,
    RedisRateLimiter,
    SemanticCacheService,
)
from src.worker.app import celery_app

logger = logging.getLogger(__name__)

_embedding_service = EmbeddingService()
_cache_service = SemanticCacheService()
_rate_limiter = RedisRateLimiter()
_mock_llm = MockLLM(_rate_limiter)


def _backoff_seconds(retries: int, base: int = 2, cap: int = 60) -> int:
    return min(cap, int(math.pow(base, retries)))


@celery_app.task(
    bind=True,
    name="prompt.process",
    max_retries=5,
    autoretry_for=(),
    retry_jitter=True,
)
def process_prompt(self: Task, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process a prompt request end-to-end."""
    start_time = time.perf_counter()
    user_id = payload["user_id"]
    prompt_id = payload["prompt_id"]
    prompt_text = payload["text"]
    priority_raw = payload.get("priority", PromptPriority.NORMAL.value)

    try:
        priority = PromptPriority(priority_raw)
    except ValueError:
        priority = PromptPriority.NORMAL

    logger.info(
        "Processing prompt",
        extra={
            "user_id": user_id,
            "prompt_id": prompt_id,
            "attempt": self.request.retries + 1,
        },
    )

    collection = get_prompt_requests_collection()
    retry_exc: Exception | None = None
    retry_delay: int | None = None
    result_payload: Dict[str, Any] | None = None

    try:
        # Find or create request (atomic operation)
        request = collection.find_one({"user_id": user_id, "prompt_id": prompt_id})

        if request is None:
            # Create request if it doesn't exist
            now = datetime.now(timezone.utc)
            request_doc = {
                "user_id": user_id,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "status": PromptStatus.RECEIVED.value,
                "priority": priority.value,
                "cached": False,
                "retry_count": 0,
                "processing_time_ms": None,
                "error_message": None,
                "cache_entry_id": None,
                "created_at": now,
                "updated_at": now,
            }
            try:
                collection.insert_one(request_doc)
                request = request_doc
            except DuplicateKeyError:
                # Another worker created it, fetch it
                request = collection.find_one({"user_id": user_id, "prompt_id": prompt_id})
                if request:
                    collection.update_one(
                        {"user_id": user_id, "prompt_id": prompt_id},
                        {
                            "$set": {
                                "prompt_text": prompt_text,
                                "priority": priority.value,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        },
                    )
                    request["prompt_text"] = prompt_text
                    request["priority"] = priority.value
        else:
            # Update existing request
            collection.update_one(
                {"user_id": user_id, "prompt_id": prompt_id},
                {
                    "$set": {
                        "prompt_text": prompt_text,
                        "priority": priority.value,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            request["prompt_text"] = prompt_text
            request["priority"] = priority.value

        # Check if already completed with cache
        if (
            request.get("status") == PromptStatus.COMPLETED.value
            and request.get("cache_entry_id")
        ):
            from src.services.qdrant_client import QdrantCacheService

            qdrant = QdrantCacheService()
            cache_point = qdrant.get(request["cache_entry_id"])
            if cache_point:
                _cache_service.record_hit(request["cache_entry_id"])
                elapsed = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "Returning cached response for completed request",
                    extra={
                        "user_id": user_id,
                        "prompt_id": prompt_id,
                    },
                )
                result_payload = _build_result(
                    user_id=user_id,
                    prompt_id=prompt_id,
                    status=PromptStatus.COMPLETED,
                    response_text=cache_point["payload"].get("response_text", ""),
                    cached=True,
                    retry_count=self.request.retries,
                    processing_time_ms=elapsed,
                )
                return result_payload

        # Update status to processing
        now = datetime.now(timezone.utc)
        collection.update_one(
            {"user_id": user_id, "prompt_id": prompt_id},
            {
                "$set": {
                    "status": PromptStatus.PROCESSING.value,
                    "retry_count": self.request.retries,
                    "cached": False,
                    "error_message": None,
                    "updated_at": now,
                }
            },
        )

        # Generate embedding and check cache
        embedding = _embedding_service.embed(prompt_text)
        cache_hit = _cache_service.lookup(embedding)

        if cache_hit:
            _cache_service.record_hit(cache_hit.cache_entry_id)
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            collection.update_one(
                {"user_id": user_id, "prompt_id": prompt_id},
                {
                    "$set": {
                        "status": PromptStatus.COMPLETED.value,
                        "cached": True,
                        "cache_entry_id": cache_hit.cache_entry_id,
                        "processing_time_ms": processing_time_ms,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )

            logger.info(
                "Cache hit for prompt",
                extra={
                    "user_id": user_id,
                    "prompt_id": prompt_id,
                    "similarity": cache_hit.similarity,
                },
            )
            result_payload = _build_result(
                user_id=user_id,
                prompt_id=prompt_id,
                status=PromptStatus.COMPLETED,
                response_text=cache_hit.response_text,
                cached=True,
                retry_count=self.request.retries,
                processing_time_ms=processing_time_ms,
            )
        else:
            try:
                response_text = _mock_llm.complete(prompt_text)
            except (RateLimitExceeded, ProviderError) as exc:
                new_status = (
                    PromptStatus.QUEUED.value
                    if isinstance(exc, RateLimitExceeded)
                    else PromptStatus.PROCESSING.value
                )
                collection.update_one(
                    {"user_id": user_id, "prompt_id": prompt_id},
                    {
                        "$set": {
                            "status": new_status,
                            "retry_count": self.request.retries + 1,
                            "error_message": str(exc)[:255],
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )

                logger.warning(
                    "Transient provider error; scheduling retry",
                    extra={
                        "user_id": user_id,
                        "prompt_id": prompt_id,
                        "attempt": self.request.retries + 1,
                        "error": type(exc).__name__,
                    },
                )

                retry_exc = exc
                retry_delay = _backoff_seconds(self.request.retries + 1)
            else:
                # Store in cache and mark as completed
                cache_entry_id = _cache_service.store(
                    prompt_text=prompt_text,
                    embedding=embedding,
                    response_text=response_text,
                )

                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                collection.update_one(
                    {"user_id": user_id, "prompt_id": prompt_id},
                    {
                        "$set": {
                            "status": PromptStatus.COMPLETED.value,
                            "cached": False,
                            "cache_entry_id": cache_entry_id,
                            "processing_time_ms": processing_time_ms,
                            "retry_count": self.request.retries,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )

                logger.info(
                    "Prompt processed via mock LLM",
                    extra={
                        "user_id": user_id,
                        "prompt_id": prompt_id,
                        "retry_count": self.request.retries,
                        "processing_time_ms": processing_time_ms,
                    },
                )

                result_payload = _build_result(
                    user_id=user_id,
                    prompt_id=prompt_id,
                    status=PromptStatus.COMPLETED,
                    response_text=response_text,
                    cached=False,
                    retry_count=self.request.retries,
                    processing_time_ms=processing_time_ms,
                )

    except Exception as exc:
        logger.exception(
            "Unexpected error during prompt processing",
            extra={
                "user_id": user_id,
                "prompt_id": prompt_id,
            },
        )
        _mark_failed(user_id, prompt_id, str(exc))
        raise

    if retry_exc is not None and retry_delay is not None:
        raise self.retry(exc=retry_exc, countdown=retry_delay)

    if result_payload is None:
        logger.error(
            "Processing completed without result or retry instruction",
            extra={
                "user_id": user_id,
                "prompt_id": prompt_id,
            },
        )
        raise RuntimeError("No result produced for prompt processing.")

    return result_payload


def _build_result(
    *,
    user_id: str,
    prompt_id: str,
    status: PromptStatus,
    response_text: str,
    cached: bool,
    retry_count: int,
    processing_time_ms: int,
) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "prompt_id": prompt_id,
        "status": status.value,
        "cached": cached,
        "response": response_text,
        "retry_count": retry_count,
        "processing_time_ms": processing_time_ms,
    }


def _mark_failed(user_id: str, prompt_id: str, error_message: str) -> None:
    collection = get_prompt_requests_collection()
    request = collection.find_one({"user_id": user_id, "prompt_id": prompt_id})
    current_retry_count = request.get("retry_count", 0) if request else 0
    
    collection.update_one(
        {"user_id": user_id, "prompt_id": prompt_id},
        {
            "$set": {
                "status": PromptStatus.FAILED.value,
                "error_message": error_message[:255],
                "cached": False,
                "processing_time_ms": None,
                "retry_count": current_retry_count + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
