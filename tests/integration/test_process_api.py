from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import (
    get_cache_service,
    get_db_session,
    get_embedding_service,
    get_prompt_task_client,
)
from src.models import PromptPriority, PromptRequest, PromptStatus
from src.services.semantic_cache import CacheHit


class FakeSession:
    def __init__(self) -> None:
        self._execute_responses: List[Optional[PromptRequest]] = []
        self.cache_entries: Dict[int, object] = {}
        self.added: List[object] = []
        self.flush_called = 0

    def queue_execute(self, response: Optional[PromptRequest]) -> None:
        self._execute_responses.append(response)

    def execute(self, _statement):
        response = (
            self._execute_responses.pop(0)
            if self._execute_responses
            else None
        )

        class _Result:
            def __init__(self, obj):
                self.obj = obj

            def scalar_one_or_none(self):
                return self.obj

            def first(self):
                return self.obj

        return _Result(response)

    def get(self, model, pk):
        return self.cache_entries.get(pk)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.flush_called += 1

    def commit(self):
        return None

    def refresh(self, _obj):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


class StubEmbeddingService:
    def embed(self, text: str):
        return [0.1] * 768


class StubCacheService:
    def __init__(self, hit: Optional[CacheHit] = None):
        self.hit = hit
        self.recorded_hits: List[int] = []

    def lookup(self, _session, _embedding, include_stale: bool = False):
        return self.hit

    def record_hit(self, _session, cache_entry_id: int):
        self.recorded_hits.append(cache_entry_id)


class StubResult:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def ready(self) -> bool:
        return True

    def failed(self) -> bool:
        return False

    @property
    def result(self) -> dict[str, object]:
        return self._payload

    def get(self, timeout=None, propagate: bool = True):
        if isinstance(self._payload, Exception):
            if propagate:
                raise self._payload
            return self._payload
        return self._payload


class StubTaskClient:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload
        self.enqueued: List[dict[str, object]] = []

    def enqueue(self, **task_kwargs):
        self.enqueued.append(task_kwargs)
        return StubResult(self.payload)


@pytest.fixture()
def client():
    original_overrides = app.dependency_overrides.copy()
    session = FakeSession()
    session.cache_entries[1] = type(
        "CacheEntry",
        (),
        {"id": 1, "response_text": "Cached answer"},
    )()

    cache_hit = CacheHit(
        cache_entry_id=1,
        response_text="Cached answer",
        similarity=0.95,
        created_at=datetime.now(timezone.utc),
        hit_count=0,
    )
    cache_service = StubCacheService(hit=cache_hit)
    task_client = StubTaskClient(
        {
            "user_id": "u1",
            "prompt_id": "p-miss",
            "status": "completed",
            "cached": False,
            "response": "Fresh answer",
            "retry_count": 0,
            "processing_time_ms": 123,
        }
    )

    def override_session():
        yield session

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_embedding_service] = lambda: StubEmbeddingService()
    app.dependency_overrides[get_cache_service] = lambda: cache_service
    app.dependency_overrides[get_prompt_task_client] = lambda: task_client

    with TestClient(app) as test_client:
        yield test_client, session, cache_service, task_client

    app.dependency_overrides = original_overrides


def test_process_returns_cached_response(client):
    test_client, session, cache_service, _ = client

    completed_request = PromptRequest(
        user_id="u1",
        prompt_id="p-cache",
        prompt_text="Explain cats",
        priority=PromptPriority.NORMAL,
        status=PromptStatus.COMPLETED,
    )
    completed_request.cache_entry_id = 1
    completed_request.retry_count = 1

    session.queue_execute(completed_request)  # existing request lookup

    response = test_client.post(
        "/process",
        json={
            "user_id": "u1",
            "prompt_id": "p-cache",
            "text": "Explain cats",
            "priority": "normal",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["cached"] is True
    assert body["response"] == "Cached answer"
    assert cache_service.recorded_hits == [1]


def test_process_enqueues_on_cache_miss(client):
    test_client, session, cache_service, task_client = client

    # Cache lookup should miss for this request.
    cache_service.hit = None
    session.queue_execute(None)  # no existing request
    session.queue_execute(None)  # during persist helper

    response = test_client.post(
        "/process",
        json={
            "user_id": "u1",
            "prompt_id": "p-miss",
            "text": "Explain dogs",
            "priority": "high",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["cached"] is False
    assert body["response"] == "Fresh answer"
    assert task_client.enqueued


def test_get_status_returns_completed_payload(client):
    test_client, session, cache_service, _ = client

    completed_request = PromptRequest(
        user_id="u-status",
        prompt_id="p-status",
        prompt_text="Explain cats",
        priority=PromptPriority.NORMAL,
        status=PromptStatus.COMPLETED,
    )
    completed_request.cache_entry_id = 1
    completed_request.retry_count = 2
    completed_request.processing_time_ms = 1234
    completed_request.cached = True

    session.queue_execute(completed_request)

    response = test_client.get("/process/u-status/p-status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["cached"] is True
    assert body["retry_count"] == 2
    assert body["processing_time_ms"] == 1234


def test_metrics_endpoint_exposes_counters(client):
    test_client, _, _, _ = client

    # Hit metrics endpoint and ensure Prometheus text format is returned.
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "semantic_cache_hits_total" in response.text


def test_task_client_priority_queue_mapping():
    from src.api.services import PromptTaskClient

    client = PromptTaskClient()
    assert client.PRIORITY_TO_QUEUE["high"] == "prompt_high"
    assert client.PRIORITY_TO_QUEUE["normal"] == "prompt_normal"
    assert client.PRIORITY_TO_QUEUE["low"] == "prompt_low"

