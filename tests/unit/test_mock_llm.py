from __future__ import annotations

import pytest

from src.services.exceptions import ProviderError, RateLimitExceeded
from src.services.mock_llm import MockLLM


class StubLimiter:
    def __init__(self, should_raise: bool = False):
        self.should_raise = should_raise
        self.acquired = 0

    def acquire(self):
        if self.should_raise:
            raise RateLimitExceeded("limit hit")
        self.acquired += 1
        return self

    def release(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_mock_llm_success(monkeypatch) -> None:
    limiter = StubLimiter()
    llm = MockLLM(
        limiter,
        min_latency_seconds=0.0,
        max_latency_seconds=0.0,
        failure_rate=0.0,
    )

    response = llm.complete("Explain AI in simple terms.")

    assert "Explain AI in simple terms."[:20] in response
    assert limiter.acquired == 1


def test_mock_llm_rate_limit(monkeypatch) -> None:
    limiter = StubLimiter(should_raise=True)
    llm = MockLLM(
        limiter,
        min_latency_seconds=0.0,
        max_latency_seconds=0.0,
        failure_rate=0.0,
    )

    with pytest.raises(RateLimitExceeded):
        llm.complete("prompt")


def test_mock_llm_provider_failure(monkeypatch) -> None:
    limiter = StubLimiter()
    llm = MockLLM(
        limiter,
        min_latency_seconds=0.0,
        max_latency_seconds=0.0,
        failure_rate=0.99,
    )

    monkeypatch.setattr("random.random", lambda: 0.0)

    with pytest.raises(ProviderError):
        llm.complete("prompt")


class CountingLimiter:
    def __init__(self, limit: int):
        self.limit = limit
        self.calls = 0

    def acquire(self):
        if self.calls >= self.limit:
            raise RateLimitExceeded("limit hit")
        self.calls += 1

        class Token:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        return Token()


def test_mock_llm_rate_limit_enforced() -> None:
    limiter = CountingLimiter(limit=3)
    llm = MockLLM(
        limiter,
        min_latency_seconds=0.0,
        max_latency_seconds=0.0,
        failure_rate=0.0,
    )

    for _ in range(3):
        llm.complete("prompt")

    with pytest.raises(RateLimitExceeded):
        llm.complete("prompt beyond limit")

