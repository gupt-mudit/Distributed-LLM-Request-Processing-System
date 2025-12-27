from __future__ import annotations

import hashlib
import os
import threading
from typing import Iterable, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None  # type: ignore[assignment]


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))


class EmbeddingService:
    """Generates embeddings using sentence-transformers or a deterministic fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        deterministic_fallback: bool | None = None,
    ) -> None:
        self._model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            DEFAULT_MODEL_NAME,
        )
        # Auto-enable deterministic mode if sentence-transformers is missing.
        if deterministic_fallback is None:
            deterministic_fallback = SentenceTransformer is None
        self._deterministic = deterministic_fallback
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def embed(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            raise ValueError("Text to embed cannot be empty.")

        if self._deterministic:
            return self._deterministic_embedding(text)

        model = self._get_model()
        embedding = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return embedding.astype(np.float32).tolist()

    # ---- internal helpers -------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not available; enable deterministic fallback."
            )

        with self._lock:
            if self._model is None:
                self._model = SentenceTransformer(self._model_name)
        return self._model

    def _deterministic_embedding(self, text: str) -> List[float]:
        """Produce a repeatable pseudo-embedding for testing."""
        # Use SHA256 to derive a deterministic seed.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.normal(0, 1, EMBEDDING_DIMENSION)
        vector = vector / np.linalg.norm(vector)
        return vector.astype(np.float32).tolist()


def embed_batch(service: EmbeddingService, texts: Iterable[str]) -> List[List[float]]:
    return [service.embed(text) for text in texts]

