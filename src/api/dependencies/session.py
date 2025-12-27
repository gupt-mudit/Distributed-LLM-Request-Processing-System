from __future__ import annotations

from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from src.models.database import get_session_factory


def get_db_session() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

