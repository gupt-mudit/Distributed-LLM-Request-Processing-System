from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.models.prompt_cache_entry import EMBEDDING_DIMENSION, PromptCacheEntry
from src.services.semantic_cache import CacheHit, SemanticCacheService


def _make_entry(entry_id: int, response: str, hit_count: int = 0) -> PromptCacheEntry:
    entry = PromptCacheEntry(
        prompt_text="cached prompt",
        embedding=[0.0] * 768,
        response_text=response,
    )
    entry.id = entry_id  # type: ignore[attr-defined]
    entry.hit_count = hit_count
    entry.created_at = datetime.now(timezone.utc)
    return entry


def test_lookup_returns_none_when_no_result() -> None:
    session = MagicMock()
    session.execute().first.return_value = None

    service = SemanticCacheService(similarity_threshold=0.9)

    result = service.lookup(session, [0.0] * EMBEDDING_DIMENSION)

    assert result is None


def test_lookup_returns_hit_when_similarity_high() -> None:
    entry = _make_entry(1, "cached response")
    session = MagicMock()
    session.execute().first.return_value = (entry, 0.95)

    service = SemanticCacheService(similarity_threshold=0.9)

    result = service.lookup(session, [0.1] * EMBEDDING_DIMENSION)

    assert isinstance(result, CacheHit)
    assert result.cache_entry_id == 1
    assert result.response_text == "cached response"


def test_lookup_returns_none_when_similarity_low() -> None:
    entry = _make_entry(1, "cached response")
    session = MagicMock()
    session.execute().first.return_value = (entry, 0.5)

    service = SemanticCacheService(similarity_threshold=0.9)

    result = service.lookup(session, [0.1] * EMBEDDING_DIMENSION)

    assert result is None


def test_record_hit_updates_entry() -> None:
    entry = _make_entry(1, "cached")
    session = MagicMock()
    session.execute().scalar_one_or_none.return_value = entry

    service = SemanticCacheService()
    service.record_hit(session, 1)

    assert entry.hit_count == 1
    assert entry.last_hit_at is not None
    session.add.assert_called_once_with(entry)


def test_store_persists_entry() -> None:
    session = MagicMock()

    service = SemanticCacheService()
    entry = service.store(
        session=session,
        prompt_text="hello",
        embedding=[0.1] * EMBEDDING_DIMENSION,
        response_text="world",
    )

    session.add.assert_called_once_with(entry)
    session.flush.assert_called_once()


def test_validate_embedding_dimension_mismatch() -> None:
    service = SemanticCacheService()
    with pytest.raises(ValueError):
        service.lookup(MagicMock(), [0.0])

