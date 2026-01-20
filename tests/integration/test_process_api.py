from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import (
    get_cache_service,
    get_db_collection,
    get_embedding_service,
    get_prompt_task_client,
)
from src.models.mongodb_models import PromptPriority, PromptStatus
from src.services.qdrant_client import QdrantCacheService
from src.services.semantic_cache import CacheHit


class FakeCollection:
    """Mock MongoDB collection for testing."""
    
    def __init__(self) -> None:
        self._documents: Dict[tuple[str, str], dict] = {}
        self._find_one_responses: List[Optional[dict]] = []
        self._update_calls: List[dict] = []
    
    def queue_find_one(self, response: Optional[dict]) -> None:
        """Queue a response for find_one calls."""
        self._find_one_responses.append(response)
    
    def find_one(self, filter_dict: dict) -> Optional[dict]:
        """Mock find_one - returns queued response or document from _documents."""
        if self._find_one_responses:
            return self._find_one_responses.pop(0)
        
        # Check if document exists in _documents
        user_id = filter_dict.get("user_id")
        prompt_id = filter_dict.get("prompt_id")
        if user_id and prompt_id:
            return self._documents.get((user_id, prompt_id))
        return None
    
    def update_one(self, filter_dict: dict, update_dict: dict, upsert: bool = False) -> None:
        """Mock update_one - stores document in _documents."""
        self._update_calls.append({
            "filter": filter_dict,
            "update": update_dict,
            "upsert": upsert,
        })
        
        user_id = filter_dict.get("user_id")
        prompt_id = filter_dict.get("prompt_id")
        if user_id and prompt_id:
            key = (user_id, prompt_id)
            if key not in self._documents:
                self._documents[key] = {
                    "user_id": user_id,
                    "prompt_id": prompt_id,
                    "created_at": datetime.now(timezone.utc),
                }
            
            # Apply $set updates
            if "$set" in update_dict:
                self._documents[key].update(update_dict["$set"])


class StubEmbeddingService:
    def embed(self, text: str):
        return [0.1] * 384  # Updated to 384 dimensions


class StubCacheService:
    def __init__(self, hit: Optional[CacheHit] = None):
        self.hit = hit
        self.recorded_hits: List[str] = []  # Changed to List[str] for Qdrant point IDs
    
    def lookup(self, embedding, include_stale: bool = False):
        return self.hit
    
    def record_hit(self, cache_entry_id: str):  # Changed to str
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
    collection = FakeCollection()
    
    cache_hit = CacheHit(
        cache_entry_id="point-1",  # Changed to string
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
    
    def override_collection():
        yield collection
    
    app.dependency_overrides[get_db_collection] = override_collection
    app.dependency_overrides[get_embedding_service] = lambda: StubEmbeddingService()
    app.dependency_overrides[get_cache_service] = lambda: cache_service
    app.dependency_overrides[get_prompt_task_client] = lambda: task_client
    
    with TestClient(app) as test_client:
        yield test_client, collection, cache_service, task_client
    
    app.dependency_overrides = original_overrides


def test_process_returns_cached_response(client):
    test_client, collection, cache_service, _ = client
    
    # Setup: existing completed request in MongoDB
    completed_doc = {
        "user_id": "u1",
        "prompt_id": "p-cache",
        "prompt_text": "Explain cats",
        "status": PromptStatus.COMPLETED.value,
        "priority": PromptPriority.NORMAL.value,
        "cache_entry_id": "point-1",
        "retry_count": 1,
        "cached": True,
    }
    collection._documents[("u1", "p-cache")] = completed_doc
    
    # Mock Qdrant to return cache entry
    with patch.object(QdrantCacheService, 'get') as mock_qdrant_get:
        mock_qdrant_get.return_value = {
            "id": "point-1",
            "payload": {"response_text": "Cached answer"}
        }
        
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
    assert cache_service.recorded_hits == ["point-1"]


def test_process_enqueues_on_cache_miss(client):
    test_client, collection, cache_service, task_client = client
    
    # Cache lookup should miss for this request.
    cache_service.hit = None
    collection.queue_find_one(None)  # no existing request
    
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
    assert body["status"] == "queued"  # Changed: returns immediately with queued
    assert task_client.enqueued


def test_get_status_returns_completed_payload(client):
    test_client, collection, cache_service, _ = client
    
    # Setup: completed request in MongoDB
    completed_doc = {
        "user_id": "u-status",
        "prompt_id": "p-status",
        "prompt_text": "Explain cats",
        "status": PromptStatus.COMPLETED.value,
        "priority": PromptPriority.NORMAL.value,
        "cache_entry_id": "point-1",
        "retry_count": 2,
        "processing_time_ms": 1234,
        "cached": True,
    }
    collection._documents[("u-status", "p-status")] = completed_doc
    
    # Mock Qdrant to return cache entry
    with patch.object(QdrantCacheService, 'get') as mock_qdrant_get:
        mock_qdrant_get.return_value = {
            "id": "point-1",
            "payload": {"response_text": "Cached answer"}
        }
        
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
