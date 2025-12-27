from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .base import Base, create_db_engine, create_session_factory, get_session

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "scoped_session",
]


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_db_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return create_session_factory(get_engine())


@contextmanager
def scoped_session() -> Generator[Session, None, None]:
    with get_session(get_engine()) as session:
        yield session

