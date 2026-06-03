"""
session.py  —  /api/session/* blueprint (build-order step 7).

    POST /api/session/save     { exercise, reps, form_errors, duration_sec, avg_* }
                               -> { session_id }
    GET  /api/session/history  -> [ { session_id, date, exercise, reps, top_errors, ... } ]
"""
from flask import Blueprint, jsonify, request

from db.store import recent_sessions, save_session

session_bp = Blueprint("session", __name__, url_prefix="/api/session")

VALID_EXERCISES = {"squat", "deadlift"}


def _ok(data):
    return jsonify({"data": data, "error": None})


def _err(message, status=400):
    return jsonify({"data": None, "error": message}), status


@session_bp.post("/save")
def save():
    body = request.get_json(silent=True) or {}
    if body.get("exercise") not in VALID_EXERCISES:
        return _err(f"'exercise' must be one of {sorted(VALID_EXERCISES)}.")
    try:
        session_id = save_session(body)
    except Exception as exc:                       # noqa: BLE001
        return _err(f"Could not save session: {exc}", 500)
    return _ok({"session_id": session_id})


@session_bp.get("/history")
def history():
    return _ok(recent_sessions())
