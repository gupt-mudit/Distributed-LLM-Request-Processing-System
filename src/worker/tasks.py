from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict

from celery import Task
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.models import (
    PromptCacheEntry,
    PromptPriority,
    PromptRequest,
    PromptStatus,
)
from src.services import (
    EmbeddingService,
    MockLLM,
    ProviderError,
    RateLimitExceeded,
    RedisRateLimiter,
    SemanticCacheService,
)
from src.worker.app import celery_app
from src.worker.db import session_scope

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

    retry_exc: Exception | None = None
    retry_delay: int | None = None
    result_payload: Dict[str, Any] | None = None

    try:
        with session_scope() as session:
            request = (
                session.execute(
                    select(PromptRequest)
                    .where(
                        PromptRequest.user_id == user_id,
                        PromptRequest.prompt_id == prompt_id,
                    )
                    .with_for_update()
                ).scalar_one_or_none()
            )

            if request is None:
                # Create a request record if it does not already exist.
                request = PromptRequest(
                    user_id=user_id,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    priority=priority,
                    status=PromptStatus.RECEIVED,
                )
                session.add(request)
                try:
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    request = (
                        session.execute(
                            select(PromptRequest)
                            .where(
                                PromptRequest.user_id == user_id,
                                PromptRequest.prompt_id == prompt_id,
                            )
                            .with_for_update()
                        ).scalar_one()
                    )
                    request.prompt_text = prompt_text
                    request.priority = priority
            else:
                request.prompt_text = prompt_text
                request.priority = priority

            if request.status == PromptStatus.COMPLETED and request.cache_entry_id:
                cache_entry = session.get(PromptCacheEntry, request.cache_entry_id)
                if cache_entry:
                    _cache_service.record_hit(session, cache_entry.id)
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
                        status=request.status,
                        response_text=cache_entry.response_text,
                        cached=True,
                        retry_count=self.request.retries,
                        processing_time_ms=elapsed,
                    )
                    return result_payload

            request.status = PromptStatus.PROCESSING
            request.retry_count = self.request.retries
            request.cached = False
            request.error_message = None

            embedding = _embedding_service.embed(prompt_text)
            cache_hit = _cache_service.lookup(session, embedding)

            if cache_hit:
                _cache_service.record_hit(session, cache_hit.cache_entry_id)
                request.status = PromptStatus.COMPLETED
                request.cached = True
                request.cache_entry_id = cache_hit.cache_entry_id
                request.processing_time_ms = int((time.perf_counter() - start_time) * 1000)

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
                    status=request.status,
                    response_text=cache_hit.response_text,
                    cached=True,
                    retry_count=self.request.retries,
                    processing_time_ms=request.processing_time_ms,
                )
            else:
                try:
                    response_text = _mock_llm.complete(prompt_text)
                except (RateLimitExceeded, ProviderError) as exc:
                    if isinstance(exc, RateLimitExceeded):
                        request.status = PromptStatus.QUEUED
                    else:
                        request.status = PromptStatus.PROCESSING

                    request.retry_count = self.request.retries + 1
                    request.error_message = str(exc)

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
                    cache_entry = _cache_service.store(
                        session=session,
                        prompt_text=prompt_text,
                        embedding=embedding,
                        response_text=response_text,
                    )

                    request.status = PromptStatus.COMPLETED
                    request.cached = False
                    request.cache_entry_id = cache_entry.id
                    request.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                    request.retry_count = self.request.retries

                    logger.info(
                        "Prompt processed via mock LLM",
                        extra={
                            "user_id": user_id,
                            "prompt_id": prompt_id,
                            "retry_count": request.retry_count,
                            "processing_time_ms": request.processing_time_ms,
                        },
                    )

                    result_payload = _build_result(
                        user_id=user_id,
                        prompt_id=prompt_id,
                        status=request.status,
                        response_text=response_text,
                        cached=False,
                        retry_count=request.retry_count,
                        processing_time_ms=request.processing_time_ms,
                    )

    except SQLAlchemyError as exc:
        logger.exception(
            "Database error during prompt processing",
            extra={
                "user_id": user_id,
                "prompt_id": prompt_id,
            },
        )
        raise self.retry(exc=exc, countdown=_backoff_seconds(self.request.retries + 1))
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
    with session_scope() as session:
        request = (
            session.execute(
                select(PromptRequest)
                .where(
                    PromptRequest.user_id == user_id,
                    PromptRequest.prompt_id == prompt_id,
                )
                .with_for_update()
            ).scalar_one_or_none()
        )
        if request is None:
            return

        request.status = PromptStatus.FAILED
        request.error_message = error_message[:255]
        request.cached = False
        request.processing_time_ms = None
        request.retry_count = (request.retry_count or 0) + 1

