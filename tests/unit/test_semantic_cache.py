from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.services.semantic_cache import CacheHit, SemanticCacheService

EMBEDDING_DIMENSION = 384


def test_lookup_returns_none_when_no_result() -> None:
    """Test that lookup returns None when Qdrant returns no results."""
    service = SemanticCacheService(similarity_threshold=0.9)
    
    with patch.object(service._qdrant, 'search', return_value=[]):
        result = service.lookup([0.0] * EMBEDDING_DIMENSION)
        assert result is None


def test_lookup_returns_hit_when_similarity_high() -> None:
    """Test that lookup returns CacheHit when similarity is above threshold."""
    service = SemanticCacheService(similarity_threshold=0.9)
    
    mock_result = [{
        "id": "point-1",
        "score": 0.95,
        "payload": {
            "prompt_text": "cached prompt",
            "response_text": "cached response",
            "hit_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }]
    
    with patch.object(service._qdrant, 'search', return_value=mock_result):
        result = service.lookup([0.1] * EMBEDDING_DIMENSION)
        
        assert isinstance(result, CacheHit)
        assert result.cache_entry_id == "point-1"
        assert result.response_text == "cached response"
        assert result.similarity == 0.95


def test_lookup_returns_none_when_similarity_low() -> None:
    """Test that lookup returns None when similarity is below threshold."""
    service = SemanticCacheService(similarity_threshold=0.9)
    
    mock_result = [{
        "id": "point-1",
        "score": 0.5,  # Below threshold
        "payload": {
            "prompt_text": "cached prompt",
            "response_text": "cached response",
            "hit_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }]
    
    with patch.object(service._qdrant, 'search', return_value=mock_result):
        result = service.lookup([0.1] * EMBEDDING_DIMENSION)
        assert result is None


def test_record_hit_updates_entry() -> None:
    """Test that record_hit calls Qdrant's update_hit_count."""
    service = SemanticCacheService()
    
    with patch.object(service._qdrant, 'update_hit_count') as mock_update:
        service.record_hit("point-1")
        mock_update.assert_called_once_with("point-1")


def test_store_persists_entry() -> None:
    """Test that store calls Qdrant's upsert and returns point ID."""
    service = SemanticCacheService()
    
    with patch.object(service._qdrant, 'upsert', return_value="point-123") as mock_upsert:
        point_id = service.store(
            prompt_text="hello",
            embedding=[0.1] * EMBEDDING_DIMENSION,
            response_text="world",
        )
        
        assert point_id == "point-123"
        mock_upsert.assert_called_once()
        call_args = mock_upsert.call_args
        assert call_args[1]["prompt_text"] == "hello"
        assert call_args[1]["response_text"] == "world"
        assert len(call_args[1]["vector"]) == EMBEDDING_DIMENSION


def test_validate_embedding_dimension_mismatch() -> None:
    """Test that lookup raises ValueError for wrong embedding dimension."""
    service = SemanticCacheService()
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        service.lookup([0.0] * 100)  # Wrong dimension
