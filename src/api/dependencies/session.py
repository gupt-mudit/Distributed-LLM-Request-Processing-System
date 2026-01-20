from __future__ import annotations

from typing import Generator

from pymongo.collection import Collection

from src.models.mongodb import get_prompt_requests_collection


def get_db_collection() -> Generator[Collection, None, None]:
    """Get MongoDB collection for prompt_requests."""
    collection = get_prompt_requests_collection()
    yield collection

