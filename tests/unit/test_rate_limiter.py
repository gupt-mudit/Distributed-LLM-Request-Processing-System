from __future__ import annotations

import pytest

from src.services.exceptions import RateLimitExceeded
from src.services.rate_limiter import RedisRateLimiter


class StubRedisClient:
    def __init__(self):
        self.bucket = 0
        self.concurrency = 0
        self.acquire_registered = False

    def register_script(self, _script_body):
        if not self.acquire_registered:
            self.acquire_registered = True

            def acquire(*, keys, args):
                limit = int(args[0])
                max_concurrent = int(args[2])

                if self.bucket >= limit:
                    return 0

                self.bucket += 1

                if max_concurrent > 0:
                    if self.concurrency >= max_concurrent:
                        self.bucket -= 1
                        return -1
                    self.concurrency += 1

                return 1

            return acquire

        def release(*, keys):
            if self.concurrency > 0:
                self.concurrency -= 1
            return 1

        return release


@pytest.fixture()
def stub_redis(mocker):
    client = StubRedisClient()
    mocker.patch("redis.Redis.from_url", return_value=client)
    return client


def test_rate_limiter_allows_within_limit(stub_redis) -> None:
    limiter = RedisRateLimiter(limit_per_minute=3, window_seconds=60, max_concurrent=0)

    for _ in range(3):
        token = limiter.acquire()
        token.release()

    with pytest.raises(RateLimitExceeded):
        limiter.acquire()


def test_rate_limiter_enforces_concurrency(stub_redis) -> None:
    limiter = RedisRateLimiter(limit_per_minute=10, window_seconds=60, max_concurrent=2)

    token1 = limiter.acquire()
    token2 = limiter.acquire()

    with pytest.raises(RateLimitExceeded):
        limiter.acquire()

    token1.release()
    limiter.acquire().release()
    token2.release()

