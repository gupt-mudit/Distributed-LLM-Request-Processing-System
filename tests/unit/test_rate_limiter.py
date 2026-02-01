from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Mock redis module before importing rate_limiter to avoid import errors
class MockRedis:
    @staticmethod
    def from_url(*args, **kwargs):
        # This will be replaced by the fixture
        pass

redis_mock = MagicMock()
redis_mock.Redis = MockRedis
sys.modules['redis'] = redis_mock

from src.services.exceptions import RateLimitExceeded
from src.services.rate_limiter import RedisRateLimiter


class StubRedisClient:
    def __init__(self):
        self.tokens = None  # None means uninitialized, will be set to capacity on first call
        self.last_refill = None
        self.acquire_registered = False

    def register_script(self, _script_body):
        if not self.acquire_registered:
            self.acquire_registered = True

            def acquire(*, keys, args):
                # Token bucket args: capacity, refill_rate, current_time, expiration_ms
                capacity = float(args[0])
                refill_rate = float(args[1])
                current_time = int(args[2])

                # Initialize bucket if needed
                if self.tokens is None:
                    self.tokens = capacity
                    self.last_refill = current_time
                else:
                    # Refill tokens based on elapsed time
                    elapsed = current_time - self.last_refill
                    if elapsed > 0:
                        tokens_to_add = elapsed * refill_rate
                        self.tokens = min(capacity, self.tokens + tokens_to_add)
                        self.last_refill = current_time

                # Check if we have tokens
                if self.tokens < 1:
                    wait_seconds = max(1, int((1 - self.tokens) / refill_rate))
                    return [0, wait_seconds]  # {rejected, wait_time}

                # Consume one token
                self.tokens -= 1

                return [1, 0]  # {allowed, wait_time}

            return acquire

        return lambda *args, **kwargs: None  # No-op for release script


@pytest.fixture()
def stub_redis():
    client = StubRedisClient()
    # Replace the Redis.from_url method with our stub
    redis_mock.Redis.from_url = lambda *args, **kwargs: client
    return client


def test_rate_limiter_allows_within_limit(stub_redis) -> None:
    limiter = RedisRateLimiter(limit_per_minute=3, window_seconds=60)

    for _ in range(3):
        limiter.acquire()

    with pytest.raises(RateLimitExceeded):
        limiter.acquire()


def test_rate_limiter_token_refill(stub_redis) -> None:
    """Test that tokens refill over time in token bucket implementation."""
    import time
    
    limiter = RedisRateLimiter(limit_per_minute=60, window_seconds=60)
    # Refill rate = 60/60 = 1 token per second
    
    # Consume all tokens
    for _ in range(60):
        limiter.acquire()
    
    # Should be rate limited now
    with pytest.raises(RateLimitExceeded):
        limiter.acquire()
    
    # Simulate time passing: update last_refill to be 2 seconds ago
    # This simulates 2 seconds passing, which should refill 2 tokens
    current_time = int(time.time())
    stub_redis.last_refill = current_time - 2
    
    # The next acquire will use current_time, so elapsed = 2 seconds
    # This should refill 2 tokens (1 token/second * 2 seconds)
    limiter.acquire()
    limiter.acquire()
    
    # But not a third (only 2 tokens were refilled)
    with pytest.raises(RateLimitExceeded):
        limiter.acquire()

