"""
models.py  —  ORM models: Session (a workout) and RepData (optional per-rep rows).

A Session stores the aggregate produced by the analyzer's summary(): reps, the
form-error histogram, and average joint angles. RepData is defined for future
per-rep breakdowns (not populated by the current save endpoint).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    exercise: Mapped[str] = mapped_column(String(20))
    reps: Mapped[int] = mapped_column(default=0)
    duration_sec: Mapped[float] = mapped_column(default=0.0)
    form_errors: Mapped[dict] = mapped_column(JSON, default=dict)  # {error: count}
    avg_knee_angle: Mapped[float | None] = mapped_column(default=None)
    avg_hip_angle: Mapped[float | None] = mapped_column(default=None)
    avg_spine_angle: Mapped[float | None] = mapped_column(default=None)

    reps_data: Mapped[list["RepData"]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
    )

    def top_errors(self, n: int = 2) -> list[str]:
        errs = self.form_errors or {}
        if isinstance(errs, dict):
            return [k for k, _ in sorted(errs.items(), key=lambda kv: kv[1], reverse=True)[:n]]
        if isinstance(errs, list):
            return list(errs)[:n]
        return []

    def to_dict(self) -> dict:
        return {
            "session_id": self.id,
            "date": self.created_at.isoformat(),
            "exercise": self.exercise,
            "reps": self.reps,
            "duration_sec": self.duration_sec,
            "form_errors": self.form_errors or {},
            "avg_knee_angle": self.avg_knee_angle,
            "avg_hip_angle": self.avg_hip_angle,
            "avg_spine_angle": self.avg_spine_angle,
            "top_errors": self.top_errors(),
        }


class RepData(Base):
    __tablename__ = "rep_data"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    rep_index: Mapped[int] = mapped_column(default=0)
    phase: Mapped[str | None] = mapped_column(String(10), default=None)
    form_class: Mapped[str | None] = mapped_column(String(30), default=None)
    knee_angle: Mapped[float | None] = mapped_column(default=None)
    hip_angle: Mapped[float | None] = mapped_column(default=None)
    spine_angle: Mapped[float | None] = mapped_column(default=None)

    session: Mapped["Session"] = relationship(back_populates="reps_data")

    def to_dict(self) -> dict:
        return {
            "rep_index": self.rep_index,
            "phase": self.phase,
            "form_class": self.form_class,
            "knee_angle": self.knee_angle,
            "hip_angle": self.hip_angle,
            "spine_angle": self.spine_angle,
        }
