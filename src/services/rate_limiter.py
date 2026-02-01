from __future__ import annotations

import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Optional

import redis

from .exceptions import RateLimitExceeded


DEFAULT_LIMIT_PER_MINUTE = 300
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_CONCURRENCY_LIMIT = 5


class RedisRateLimiter:
    """Token-bucket rate limiter with optional concurrency control.
    """

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
        
        # Token bucket parameters
        self._capacity = limit_per_minute  # Max tokens (bucket size)
        self._refill_rate = limit_per_minute / window_seconds  # Tokens per second

        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis.Redis.from_url(redis_url)

        # Lua script for token bucket with atomic operations.
        # Uses Redis hash to store: tokens (current count) and last_refill (timestamp)
        self._acquire_script = self._client.register_script(
            """
            local bucket_key = KEYS[1]
            local concurrency_key = KEYS[2]
            local capacity = tonumber(ARGV[1])
            local refill_rate = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            local max_concurrent = tonumber(ARGV[4])
            local expiration_ms = tonumber(ARGV[5])

            -- Get or initialize bucket state (stored as Redis hash)
            local bucket_data = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
            local tokens = bucket_data[1]
            local last_refill = bucket_data[2]

            if not tokens then
                -- First request: initialize bucket with full capacity
                tokens = capacity
                last_refill = current_time
                redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', last_refill)
                redis.call('PEXPIRE', bucket_key, expiration_ms)
            else
                tokens = tonumber(tokens)
                last_refill = tonumber(last_refill)
                
                -- Calculate elapsed time and refill tokens
                local elapsed = current_time - last_refill
                if elapsed > 0 then
                    -- Refill tokens based on elapsed time (capped at capacity)
                    local tokens_to_add = elapsed * refill_rate
                    tokens = math.min(capacity, tokens + tokens_to_add)
                    last_refill = current_time
                end
            end

            -- Check if we have enough tokens
            if tokens < 1 then
                -- Calculate wait time until next token is available
                local wait_seconds = math.ceil((1 - tokens) / refill_rate)
                return {0, wait_seconds}  -- {rejected, wait_time_seconds}
            end

            -- Consume one token
            tokens = tokens - 1

            -- Update bucket state
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', last_refill)
            redis.call('PEXPIRE', bucket_key, expiration_ms)

            -- Concurrency check (same logic as before)
            if max_concurrent > 0 then
                local concurrent = redis.call('GET', concurrency_key)
                if concurrent and tonumber(concurrent) >= max_concurrent then
                    -- Rollback: refund the token we just consumed
                    tokens = tokens + 1
                    redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', last_refill)
                    return {-1, 0}  -- {concurrency_limit, wait_time}
                end
                redis.call('INCR', concurrency_key)
                redis.call('PEXPIRE', concurrency_key, expiration_ms)
            end

            return {1, 0}  -- {allowed, wait_time}
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
        """Attempt to reserve capacity; raise if the quota is exhausted.
        
        Returns a RateLimitToken that must be released (or used as context manager).
        """
        bucket_key = self._bucket_key()
        concurrency_key = self._concurrency_key()
        current_time = int(time.time())  # Unix timestamp in seconds
        expiration_ms = (self._window + 10) * 1000  # Add 10s buffer for expiration

        result = self._acquire_script(
            keys=[bucket_key, concurrency_key],
            args=[
                self._capacity,
                self._refill_rate,
                current_time,
                self._max_concurrent,
                expiration_ms,
            ],
        )

        # Result is a list: [status, wait_time]
        status = result[0]
        wait_time = result[1] if len(result) > 1 else 0

        if status == 0:
            error_msg = f"LLM rate limit exceeded ({self._limit} calls per {self._window}s)."
            if wait_time > 0:
                error_msg += f" Retry in {wait_time}s."
            raise RateLimitExceeded(error_msg)
        if status == -1:
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

