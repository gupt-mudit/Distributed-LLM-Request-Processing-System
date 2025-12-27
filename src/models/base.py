from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import MetaData, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for all ORM models with shared metadata."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


def build_database_url() -> str:
    if url := os.getenv("DATABASE_URL"):
        return url

    user = os.getenv("POSTGRES_USER", "prompt_user")
    password = os.getenv("POSTGRES_PASSWORD", "prompt_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "prompt_db")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def create_db_engine() -> Engine:
    return create_engine(
        build_database_url(),
        pool_pre_ping=True,
        future=True,
    )


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    engine = engine or create_db_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    session_factory = create_session_factory(engine)
    with session_factory() as session:
        yield session

