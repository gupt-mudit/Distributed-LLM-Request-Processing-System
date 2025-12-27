from __future__ import annotations

import sys
from typing import NoReturn

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.services.exceptions import RateLimitExceeded
from src.services.mock_llm import MockLLM
from src.services.rate_limiter import RedisRateLimiter


def main() -> NoReturn:
    limit = 5
    limiter = RedisRateLimiter(
        limit_per_minute=limit,
        window_seconds=60,
        namespace="rate-limit-demo",
    )
    llm = MockLLM(
        limiter,
        min_latency_seconds=0.0,
        max_latency_seconds=0.0,
        failure_rate=0.0,
    )

    allowed = 0
    for call_number in range(1, limit + 2):
        prompt = f"Rate limit demo call #{call_number}"
        try:
            llm.complete(prompt)
            allowed += 1
            print(f"Call {call_number}: allowed")
        except RateLimitExceeded:
            print(f"Call {call_number}: rate limit exceeded")
            break

    print(f"Total calls allowed before limit: {allowed}/{limit}")
    sys.exit(0)


if __name__ == "__main__":
    main()

