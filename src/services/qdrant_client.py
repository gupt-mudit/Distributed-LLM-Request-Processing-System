from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))


class QdrantCacheService:
    """Qdrant client wrapper for vector similarity search."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection: str = "prompt_cache",
        embedding_dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        self._url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._collection = collection
        self._embedding_dimension = embedding_dimension
        self._client = QdrantClient(url=self._url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist.
        
        Uses idempotent pattern: try to create, ignore if already exists.
        Handles race condition where multiple processes try to create
        the same collection simultaneously.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse
        
        try:
            # Try to get collection first (fast path)
            self._client.get_collection(self._collection)
            return
        except UnexpectedResponse as e:
            if e.status_code != 404:
                raise  # Re-raise non-404 errors (auth, server errors, etc.)
        except Exception:
            # Handle non-HTTP exceptions (connection errors, etc.)
            # Re-raise to avoid hiding real errors
            raise
        
        # Collection doesn't exist, create it
        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
        except UnexpectedResponse as e:
            # If collection already exists (race condition), that's fine
            if e.status_code == 409:
                # Another process created it - verify it exists
                self._client.get_collection(self._collection)
                return
            # Other HTTP errors should be raised
            raise
        except Exception:
            # Handle non-HTTP exceptions (connection errors, etc.)
            raise

    def search(
        self,
        vector: Sequence[float],
        limit: int = 1,
        score_threshold: float = 0.9,
    ) -> list[dict]:
        """Search for similar vectors.

        Returns list of results with:
        - id: point ID
        - score: similarity score (1.0 = identical, 0.0 = completely different)
        - payload: metadata (prompt_text, response_text, etc.)
        """
        self._validate_embedding(vector)
        
        # Use query_points for similarity search (accepts list[float] directly)
        results = self._client.query_points(
            collection_name=self._collection,
            query=list(vector),  # Direct vector query
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        
        # Format results
        formatted_results = []
        for point in results.points:
            # Qdrant returns score (similarity) for cosine distance
            score = point.score if hasattr(point, 'score') else 0.0
            formatted_results.append({
                "id": str(point.id),
                "score": score,
                "payload": point.payload if hasattr(point, 'payload') else {},
            })
        
        return formatted_results

    def upsert(
        self,
        point_id: Optional[str],
        vector: Sequence[float],
        prompt_text: str,
        response_text: str,
    ) -> str:
        """Insert or update a point. Returns the point ID."""
        self._validate_embedding(vector)
        if point_id is None:
            point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=list(vector),
            payload={
                "prompt_text": prompt_text,
                "response_text": response_text,
                "hit_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_hit_at": None,
            },
        )
        self._client.upsert(collection_name=self._collection, points=[point])
        return point_id

    def get(self, point_id: str) -> Optional[dict]:
        """Get a point by ID."""
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        result = results[0]
        return {
            "id": result.id,
            "payload": result.payload,
        }

    def update_hit_count(self, point_id: str) -> None:
        """Increment hit count and update last_hit_at."""
        point = self.get(point_id)
        if point is None:
            return

        payload = point["payload"]
        new_payload = {
            **payload,
            "hit_count": payload.get("hit_count", 0) + 1,
            "last_hit_at": datetime.now(timezone.utc).isoformat(),
        }

        self._client.set_payload(
            collection_name=self._collection,
            payload=new_payload,
            points=[point_id],
        )

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self._embedding_dimension}, "
                f"got {len(embedding)}."
            )

