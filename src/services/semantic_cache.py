from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from src.services import metrics
from src.services.qdrant_client import QdrantCacheService
import logging

logger = logging.getLogger(__name__)
DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_LOOKBACK = timedelta(days=30)


@dataclass
class CacheHit:
    cache_entry_id: str  # Qdrant point ID (string)
    response_text: str
    similarity: float
    created_at: datetime
    hit_count: int


class SemanticCacheService:
    """Handles similarity lookup and storage in the prompt cache using Qdrant."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        lookback_window: timedelta = DEFAULT_LOOKBACK,
        qdrant_service: Optional[QdrantCacheService] = None,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1.")
        self._similarity_threshold = similarity_threshold
        self._lookback_window = lookback_window
        self._qdrant = qdrant_service or QdrantCacheService()

    # ------------------------------------------------------------------ API
    def lookup(
        self,
        embedding: Sequence[float],
        *,
        include_stale: bool = False,
    ) -> Optional[CacheHit]:
        """Lookup similar embeddings in Qdrant."""
        results = self._qdrant.search(
            vector=embedding,
            limit=1,
            score_threshold=self._similarity_threshold,
        )

        if not results:
            metrics.record_cache_miss()
            return None

        result = results[0]
        similarity = result["score"]
        payload = result["payload"]

        # Check lookback window if not including stale
        if not include_stale:
            created_at_str = payload.get("created_at")
            if created_at_str:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                cutoff = datetime.now(timezone.utc) - self._lookback_window
                if created_at < cutoff:
                    metrics.record_cache_miss()
                    logger.info(f"Cache entry is stale, created_at: {created_at}")
                    return None

        if similarity < self._similarity_threshold:
            metrics.record_cache_miss()
            logger.info(f"similarity is less than threshold, similarity: {similarity:.4f}")
            return None

        logger.info(f"similarity is greater than threshold, similarity: {similarity:.4f}")
        metrics.record_cache_hit()

        created_at_str = payload.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            if created_at_str
            else datetime.now(timezone.utc)
        )

        return CacheHit(
            cache_entry_id=result["id"],
            response_text=payload.get("response_text", ""),
            similarity=similarity,
            created_at=created_at,
            hit_count=payload.get("hit_count", 0),
        )

    def get_response_text(self, cache_entry_id: str) -> Optional[str]:
        """Get response text from a cache entry by ID.
        
        Args:
            cache_entry_id: The Qdrant point ID.
            
        Returns:
            Response text if found, None otherwise. Handles errors gracefully.
        """
        try:
            cache_point = self._qdrant.get(cache_entry_id)
            if cache_point:
                return cache_point.get("payload", {}).get("response_text")
            return None
        except Exception as exc:
            logger.warning(
                f"Failed to retrieve cache entry {cache_entry_id}: {exc}",
                exc_info=False,
            )
            return None

    def record_hit(self, cache_entry_id: str) -> None:
        """Record a cache hit (increment hit count)."""
        self._qdrant.update_hit_count(cache_entry_id)

    def store(
        self,
        prompt_text: str,
        embedding: Sequence[float],
        response_text: str,
        point_id: Optional[str] = None,
    ) -> str:
        """Store a new cache entry. Returns the point ID."""
        return self._qdrant.upsert(
            point_id=point_id,
            vector=embedding,
            prompt_text=prompt_text,
            response_text=response_text,
        )

