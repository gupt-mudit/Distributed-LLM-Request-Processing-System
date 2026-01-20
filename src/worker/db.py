from __future__ import annotations

from pymongo.collection import Collection

from src.models.mongodb import get_prompt_requests_collection


def get_collection() -> Collection:
    """Get MongoDB collection for prompt_requests."""
    return get_prompt_requests_collection()

