class RateLimitExceeded(Exception):
    """Raised when the rate limiter denies a request."""


class ProviderError(Exception):
    """Raised when the mock provider simulates a failure."""

