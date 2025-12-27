from __future__ import annotations

from src.services.embeddings import EmbeddingService


def test_embedding_service_deterministic_same_text() -> None:
    service = EmbeddingService(deterministic_fallback=True)

    vec1 = service.embed("hello world")
    vec2 = service.embed("hello world")

    assert vec1 == vec2


def test_embedding_service_deterministic_differs_for_different_text() -> None:
    service = EmbeddingService(deterministic_fallback=True)

    vec1 = service.embed("hello world")
    vec2 = service.embed("hello there")

    assert vec1 != vec2
    assert len(vec1) == len(vec2)

