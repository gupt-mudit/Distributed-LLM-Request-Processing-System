from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from src.models import PromptCacheEntry
from src.models.prompt_cache_entry import EMBEDDING_DIMENSION
from src.services import metrics

DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_LOOKBACK = timedelta(days=30)


@dataclass
class CacheHit:
    cache_entry_id: int
    response_text: str
    similarity: float
    created_at: datetime
    hit_count: int


class SemanticCacheService:
    """Handles similarity lookup and storage in the prompt cache."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        lookback_window: timedelta = DEFAULT_LOOKBACK,
        embedding_dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1.")
        self._similarity_threshold = similarity_threshold
        self._lookback_window = lookback_window
        self._embedding_dimension = embedding_dimension

    # ------------------------------------------------------------------ API
    def lookup(
        self,
        session: Session,
        embedding: Sequence[float],
        *,
        include_stale: bool = False,
    ) -> Optional[CacheHit]:
        self._validate_embedding(embedding)
        stmt = self._build_lookup_query(embedding, include_stale=include_stale)
        result = session.execute(stmt).first()
        if result is None:
            metrics.record_cache_miss()
            return None

        entry: PromptCacheEntry
        similarity: float
        entry, similarity = result

        if similarity < self._similarity_threshold:
            metrics.record_cache_miss()
            return None

        metrics.record_cache_hit()
        return CacheHit(
            cache_entry_id=entry.id,
            response_text=entry.response_text,
            similarity=similarity,
            created_at=entry.created_at,
            hit_count=entry.hit_count,
        )

    def record_hit(self, session: Session, cache_entry_id: int) -> None:
        entry = (
            session.execute(
                select(PromptCacheEntry)
                .where(PromptCacheEntry.id == cache_entry_id)
                .with_for_update()
            ).scalar_one_or_none()
        )
        if entry is None:
            return
        entry.hit_count += 1
        entry.last_hit_at = datetime.now(timezone.utc)
        session.add(entry)

    def store(
        self,
        session: Session,
        prompt_text: str,
        embedding: Sequence[float],
        response_text: str,
    ) -> PromptCacheEntry:
        self._validate_embedding(embedding)
        entry = PromptCacheEntry(
            prompt_text=prompt_text,
            embedding=list(embedding),
            response_text=response_text,
        )
        session.add(entry)
        session.flush()  # populate PK for callers
        return entry

    # ----------------------------------------------------------------- utils
    def _build_lookup_query(
        self,
        embedding: Sequence[float],
        *,
        include_stale: bool,
    ) -> Select[tuple[PromptCacheEntry, float]]:
        stmt = (
            select(
                PromptCacheEntry,
                (1 - PromptCacheEntry.embedding.cosine_distance(list(embedding))).label("similarity"),
            )
            .order_by(PromptCacheEntry.embedding.cosine_distance(list(embedding)))
            .limit(1)
        )

        if not include_stale:
            cutoff = datetime.now(timezone.utc) - self._lookback_window
            stmt = stmt.where(PromptCacheEntry.created_at >= cutoff)
        return stmt

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self._embedding_dimension}, "
                f"got {len(embedding)}."
            )

