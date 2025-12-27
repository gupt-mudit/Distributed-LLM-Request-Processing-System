from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import (
    get_cache_service,
    get_db_session,
    get_embedding_service,
    get_prompt_task_client,
)
from src.api.schemas import ProcessRequest, ProcessResponse
from src.models import PromptCacheEntry, PromptPriority, PromptRequest, PromptStatus
from src.services.semantic_cache import CacheHit

router = APIRouter(prefix="/process", tags=["processing"])

logger = logging.getLogger(__name__)


@router.post("", response_model=ProcessResponse)
async def process_prompt(
    payload: ProcessRequest,
    session: Session = Depends(get_db_session),
    embedding_service=Depends(get_embedding_service),
    cache_service=Depends(get_cache_service),
    task_client=Depends(get_prompt_task_client),
) -> ProcessResponse:
    start_time = time.perf_counter()

    existing = (
        session.execute(
            select(PromptRequest)
            .where(
                PromptRequest.user_id == payload.user_id,
                PromptRequest.prompt_id == payload.prompt_id,
            )
            .with_for_update()
        ).scalar_one_or_none()
    )

    if existing and existing.status == PromptStatus.COMPLETED and existing.cache_entry_id:
        cache_entry = session.get(PromptCacheEntry, existing.cache_entry_id)
        if cache_entry:
            cache_service.record_hit(session, cache_entry.id)
            return ProcessResponse(
                user_id=payload.user_id,
                prompt_id=payload.prompt_id,
                status=PromptStatus.COMPLETED.value,
                cached=True,
                response=cache_entry.response_text,
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                retry_count=existing.retry_count or 0,
            )

    if existing and existing.status in {PromptStatus.QUEUED, PromptStatus.PROCESSING}:
        return ProcessResponse(
            user_id=existing.user_id,
            prompt_id=existing.prompt_id,
            status=existing.status.value,
            cached=bool(existing.cached),
            response=(
                session.get(PromptCacheEntry, existing.cache_entry_id).response_text
                if existing.cache_entry_id
                else None
            ),
            processing_time_ms=existing.processing_time_ms,
            retry_count=existing.retry_count or 0,
            error=existing.error_message,
        )

    if existing and existing.status == PromptStatus.FAILED:
        return ProcessResponse(
            user_id=existing.user_id,
            prompt_id=existing.prompt_id,
            status=PromptStatus.FAILED.value,
            cached=False,
            response=None,
            processing_time_ms=existing.processing_time_ms,
            retry_count=existing.retry_count or 0,
            error=existing.error_message,
        )

    prompt_text = payload.text.strip()
    embedding = embedding_service.embed(prompt_text)
    cache_hit = cache_service.lookup(session, embedding)

    if cache_hit:
        cache_service.record_hit(session, cache_hit.cache_entry_id)
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        _persist_prompt_request(
            session,
            payload=payload,
            status=PromptStatus.COMPLETED,
            cache_entry_id=cache_hit.cache_entry_id,
            cached=True,
            processing_time_ms=processing_time_ms,
            retry_count=existing.retry_count if existing else 0,
        )
        return ProcessResponse(
            user_id=payload.user_id,
            prompt_id=payload.prompt_id,
            status=PromptStatus.COMPLETED.value,
            cached=True,
            response=cache_hit.response_text,
            processing_time_ms=processing_time_ms,
            retry_count=existing.retry_count if existing else 0,
        )

    request = existing or PromptRequest(
        user_id=payload.user_id,
        prompt_id=payload.prompt_id,
        prompt_text=prompt_text,
        priority=PromptPriority(payload.priority),
        status=PromptStatus.QUEUED,
    )
    request.prompt_text = prompt_text
    request.priority = PromptPriority(payload.priority)
    request.status = PromptStatus.QUEUED
    request.cached = False
    request.retry_count = existing.retry_count if existing else 0
    session.add(request)
    session.flush()
    session.commit()
    session.refresh(request)

    result = task_client.enqueue(
        user_id=payload.user_id,
        prompt_id=payload.prompt_id,
        text=prompt_text,
        priority=payload.priority,
    )

    task_result = _wait_for_result(result, timeout_seconds=120)

    return ProcessResponse(**task_result)


@router.get("/{user_id}/{prompt_id}", response_model=ProcessResponse)
async def get_prompt_status(
    user_id: str,
    prompt_id: str,
    session: Session = Depends(get_db_session),
) -> ProcessResponse:
    request = (
        session.execute(
            select(PromptRequest)
            .where(
                PromptRequest.user_id == user_id,
                PromptRequest.prompt_id == prompt_id,
            )
        ).scalar_one_or_none()
    )

    if request is None:
        raise HTTPException(
            status_code=404,
            detail="PromptRequest not found.",
        )

    response_text = None
    if request.cache_entry_id:
        cache_entry = session.get(PromptCacheEntry, request.cache_entry_id)
        response_text = cache_entry.response_text if cache_entry else None

    return ProcessResponse(
        user_id=request.user_id,
        prompt_id=request.prompt_id,
        status=request.status.value,
        cached=bool(request.cached),
        response=response_text,
        processing_time_ms=request.processing_time_ms,
        retry_count=request.retry_count or 0,
        error=request.error_message,
    )


def _wait_for_result(result, timeout_seconds: int = 30) -> Optional[dict[str, object]]:
    try:
        payload = result.get(timeout=timeout_seconds, propagate=False)
    except Exception as exc:  # pragma: no cover - unexpected celery failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, Exception):
        raise HTTPException(status_code=500, detail=str(payload))

    raise HTTPException(
        status_code=500,
        detail="Unexpected task result payload.",
    )


def _persist_prompt_request(
    session: Session,
    *,
    payload: ProcessRequest,
    status: PromptStatus,
    cache_entry_id: Optional[int],
    cached: bool,
    processing_time_ms: Optional[int],
    retry_count: int,
) -> None:
    request = (
        session.execute(
            select(PromptRequest)
            .where(
                PromptRequest.user_id == payload.user_id,
                PromptRequest.prompt_id == payload.prompt_id,
            )
            .with_for_update()
        ).scalar_one_or_none()
    )
    if request is None:
        request = PromptRequest(
            user_id=payload.user_id,
            prompt_id=payload.prompt_id,
            prompt_text=payload.text,
            priority=PromptPriority(payload.priority),
        )
        session.add(request)
        session.flush()

    request.status = status
    request.cache_entry_id = cache_entry_id
    request.cached = cached
    request.processing_time_ms = processing_time_ms
    request.retry_count = retry_count
    request.prompt_text = payload.text

