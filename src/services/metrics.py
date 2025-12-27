from __future__ import annotations

from prometheus_client import Counter

CACHE_HITS = Counter(
    "semantic_cache_hits_total",
    "Total number of semantic cache hits.",
)
CACHE_MISSES = Counter(
    "semantic_cache_misses_total",
    "Total number of semantic cache misses.",
)
LLM_CALLS = Counter(
    "mock_llm_calls_total",
    "Total number of mock LLM calls executed.",
)
LLM_ERRORS = Counter(
    "mock_llm_errors_total",
    "Total number of mock LLM call failures.",
)


def record_cache_hit() -> None:
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    CACHE_MISSES.inc()


def record_llm_call() -> None:
    LLM_CALLS.inc()


def record_llm_error() -> None:
    LLM_ERRORS.inc()

