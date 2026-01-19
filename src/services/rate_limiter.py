from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Optional

import redis

from .exceptions import RateLimitExceeded


DEFAULT_LIMIT_PER_MINUTE = 300
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_CONCURRENCY_LIMIT = 5


class RedisRateLimiter:
    """Token-bucket rate limiter with optional concurrency control."""

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        limit_per_minute: int = DEFAULT_LIMIT_PER_MINUTE,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        max_concurrent: int = DEFAULT_CONCURRENCY_LIMIT,
        namespace: str = "mock_llm",
    ) -> None:
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        self._limit = limit_per_minute
        self._window = window_seconds
        self._max_concurrent = max(0, max_concurrent)  # 0 disables concurrency guard
        self._namespace = namespace

        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis.Redis.from_url(redis_url)

        # Lua scripts for atomic operations.
        self._acquire_script = self._client.register_script(
            """
            local bucket_key = KEYS[1]
            local concurrency_key = KEYS[2]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local max_concurrent = tonumber(ARGV[3])

     300       local current = redis.call('GET', bucket_key)
            if current and tonumber(current) >= limit then
                return 0
            end
+300
            local new_count = redis.call('INCR', bucket_key)
            if new_count == 1 then
                redis.call('PEXPIRE', bucket_key, window * 1000)
            end

            if max_concurrent > 0 then
                local concurrent = redis.call('GET', concurrency_key)
                if concurrent and tonumber(concurrent) >= max_concurrent then
                    local reverted = redis.call('DECR', bucket_key)
                    if reverted <= 0 then
                        redis.call('DEL', bucket_key)
                    end
                    return -1
                end
                redis.call('INCR', concurrency_key)
                redis.call('PEXPIRE', concurrency_key, window * 1000)
            end

            return 1
            """
        )

        self._release_script = self._client.register_script(
            """
            local concurrency_key = KEYS[1]
            local current = redis.call('GET', concurrency_key)
            if current and tonumber(current) > 0 then
                redis.call('DECR', concurrency_key)
                if tonumber(current) <= 1 then
                    redis.call('DEL', concurrency_key)
                end
            end
            return 1
            """
        )

    # ------------------------------------------------------------------ API
    def acquire(self) -> "RateLimitToken":
        """Attempt to reserve capacity; raise if the quota is exhausted."""
        bucket_key = self._bucket_key()
        concurrency_key = self._concurrency_key()

        result = self._acquire_script(
            keys=[bucket_key, concurrency_key],
            args=[self._limit, self._window, self._max_concurrent],
        )

        if result == 0:
            raise RateLimitExceeded("LLM rate limit exceeded (300 calls per minute).")
        if result == -1:
            raise RateLimitExceeded("LLM concurrency limit reached.")

        return RateLimitToken(self, concurrency_key if self._max_concurrent > 0 else None)

    def release(self, concurrency_key: Optional[str]) -> None:
        if concurrency_key is None:
            return
        self._release_script(keys=[concurrency_key])

    # ----------------------------------------------------------------- utils
    def _bucket_key(self) -> str:
        return f"rate:{self._namespace}:bucket"

    def _concurrency_key(self) -> str:
        return f"rate:{self._namespace}:concurrent"


@dataclass
class RateLimitToken(AbstractContextManager["RateLimitToken"]):
    limiter: RedisRateLimiter
    concurrency_key: Optional[str]
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.limiter.release(self.concurrency_key)
            self.released = True

    def __enter__(self) -> "RateLimitToken":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

