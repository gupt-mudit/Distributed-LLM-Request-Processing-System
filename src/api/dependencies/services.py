from __future__ import annotations

from functools import lru_cache

from src.api.services import PromptTaskClient
from src.services import (
    EmbeddingService,
    MockLLM,
    RedisRateLimiter,
    SemanticCacheService,
)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache(maxsize=1)
def get_cache_service() -> SemanticCacheService:
    return SemanticCacheService()


@lru_cache(maxsize=1)
def get_rate_limiter() -> RedisRateLimiter:
    return RedisRateLimiter()


@lru_cache(maxsize=1)
def get_mock_llm() -> MockLLM:
    return MockLLM(get_rate_limiter())


@lru_cache(maxsize=1)
def get_prompt_task_client() -> PromptTaskClient:
    return PromptTaskClient()

