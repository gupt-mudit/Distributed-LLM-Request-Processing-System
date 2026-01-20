from __future__ import annotations

import os
from functools import lru_cache
from typing import Generator

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

def _build_mongodb_url() -> str:
    """Build MongoDB connection URL with authentication if provided."""
    url = os.getenv("MONGODB_URL")
    if url:
        return url
    
    # Build URL from components
    user = os.getenv("MONGODB_USER")
    password = os.getenv("MONGODB_PASSWORD")
    host = os.getenv("MONGODB_HOST", "localhost")
    port = os.getenv("MONGODB_PORT", "27017")
    
    if user and password:
        return f"mongodb://{user}:{password}@{host}:{port}"
    return f"mongodb://{host}:{port}"


MONGODB_URL = _build_mongodb_url()
MONGODB_DB = os.getenv("MONGODB_DB", "prompt_db")


@lru_cache(maxsize=1)
def get_mongodb_client() -> MongoClient:
    """Get MongoDB client (sync)."""
    return MongoClient(MONGODB_URL)


@lru_cache(maxsize=1)
def get_mongodb_database() -> Database:
    """Get MongoDB database."""
    client = get_mongodb_client()
    return client[MONGODB_DB]


def get_prompt_requests_collection() -> Collection:
    """Get prompt_requests collection."""
    db = get_mongodb_database()
    return db["prompt_requests"]


def init_mongodb_indexes() -> None:
    """Initialize MongoDB indexes."""
    collection = get_prompt_requests_collection()
    # Create unique index on user_id + prompt_id
    collection.create_index(
        [("user_id", 1), ("prompt_id", 1)],
        unique=True,
        name="uq_user_prompt",
    )
    # Create index on status
    collection.create_index([("status", 1)], name="idx_status")
    # Create index on created_at
    collection.create_index([("created_at", 1)], name="idx_created_at")
    # Create index on priority
    collection.create_index([("priority", 1)], name="idx_priority")


# Async client (for future async support)
@lru_cache(maxsize=1)
def get_async_mongodb_client() -> AsyncIOMotorClient:
    """Get async MongoDB client."""
    return AsyncIOMotorClient(MONGODB_URL)

