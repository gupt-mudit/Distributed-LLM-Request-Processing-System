from __future__ import annotations

import random
import time
from datetime import datetime, timezone

from .exceptions import ProviderError
from .rate_limiter import RedisRateLimiter
from . import metrics


class MockLLM:
    """Simulated LLM provider with latency, failures, and rate limiting."""

    def __init__(
        self,
        rate_limiter: RedisRateLimiter,
        *,
        min_latency_seconds: float = 0.2,
        max_latency_seconds: float = 0.5,
        failure_rate: float = 0.05,
    ) -> None:
        if min_latency_seconds < 0 or max_latency_seconds < 0:
            raise ValueError("Latency bounds must be non-negative.")
        if max_latency_seconds < min_latency_seconds:
            raise ValueError("max_latency_seconds must be >= min_latency_seconds.")
        if not (0 <= failure_rate < 1):
            raise ValueError("failure_rate must be between 0 (inclusive) and 1 (exclusive).")

        self._limiter = rate_limiter
        self._min_latency = min_latency_seconds
        self._max_latency = max_latency_seconds
        self._failure_rate = failure_rate

    def complete(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt text cannot be empty.")

        with self._limiter.acquire():
            latency = random.uniform(self._min_latency, self._max_latency)
            time.sleep(latency)

            if random.random() < self._failure_rate:
                metrics.record_llm_error()
                raise ProviderError("Mock LLM simulated provider failure.")

            metrics.record_llm_call()
            return self._format_response(prompt, latency)

    def _format_response(self, prompt: str, latency: float) -> str:
        truncated = prompt[:120]
        if len(prompt) > 120:
            truncated += "…"
        timestamp = datetime.now(timezone.utc).isoformat()
        return (
            f"[MockLLM v1 | {timestamp}] "
            f"Latency: {latency * 1000:.0f}ms. "
            f"Prompt preview: \"{truncated}\" "
            "-- This is a deterministic mock response."
        )

