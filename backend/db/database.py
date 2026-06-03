"""
database.py  —  SQLAlchemy engine / session setup (build-order step 7).

init_db() is called once at app startup. Use session_scope() for a unit of work:

    with session_scope() as s:
        s.add(obj)

SQLite needs check_same_thread=False because the Flask dev server is threaded.
expire_on_commit=False lets us read ORM attributes after the session closes
(handy for serializing to_dict right after a commit).
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_db(database_url: str) -> None:
    """Create the engine, register tables, and create them if missing."""
    global _engine, _SessionLocal
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    _engine = create_engine(database_url, connect_args=connect_args, future=True)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)

    from . import models  # noqa: F401  (import so tables register on Base.metadata)
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope():
    """Transactional scope: commit on success, rollback on error, always close."""
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized — call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
