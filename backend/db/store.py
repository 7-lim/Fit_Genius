"""
store.py  —  thin data-access helpers so API routes stay slim.

Accepts the analyzer summary dict (extra keys are ignored), persists it, and
returns plain dicts ready for JSON.
"""
from __future__ import annotations

from .database import session_scope
from .models import Session


def save_session(data: dict) -> int:
    """Persist one workout session; returns the new session id."""
    with session_scope() as s:
        row = Session(
            exercise=data["exercise"],
            reps=int(data.get("reps", 0)),
            duration_sec=float(data.get("duration_sec", 0.0)),
            form_errors=data.get("form_errors") or {},
            avg_knee_angle=data.get("avg_knee_angle"),
            avg_hip_angle=data.get("avg_hip_angle"),
            avg_spine_angle=data.get("avg_spine_angle"),
        )
        s.add(row)
        s.flush()                      # populate row.id before the session closes
        return row.id


def recent_sessions(limit: int = 50) -> list[dict]:
    """Most-recent-first list of sessions for the history view."""
    with session_scope() as s:
        rows = (
            s.query(Session)
            .order_by(Session.created_at.desc())
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in rows]


def get_session(session_id: int) -> dict | None:
    with session_scope() as s:
        row = s.get(Session, session_id)
        return row.to_dict() if row else None
