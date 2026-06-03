"""
agent.py  —  /api/agent/* blueprint (build-order step 8).

    POST /api/agent/feedback  { session_id }
                              -> { summary, corrections, tips }
    POST /api/agent/plan      { user_goals, history? }   (history defaults to stored)
                              -> { weekly_plan, focus_areas, progression_notes }

Routes stay thin: load data, call groq_service, shape the response. Heavy LLM
work lives in the service.
"""
from flask import Blueprint, jsonify, request

from db.store import get_session, recent_sessions
from services import groq_service

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


def _ok(data):
    return jsonify({"data": data, "error": None})


def _err(message, status=400):
    return jsonify({"data": None, "error": message}), status


@agent_bp.post("/feedback")
def feedback():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    if session_id is None:
        return _err("Missing 'session_id'.")
    if not groq_service.is_configured():
        return _err("AI coach unavailable — GROQ_API_KEY is not configured.", 503)

    session = get_session(int(session_id))
    if session is None:
        return _err("Session not found.", 404)

    try:
        result = groq_service.generate_session_feedback(session)
    except Exception as exc:                       # noqa: BLE001
        return _err(f"AI coach error: {exc}", 502)
    return _ok(result)


@agent_bp.post("/plan")
def plan():
    body = request.get_json(silent=True) or {}
    user_goals = body.get("user_goals") or "general strength and better lifting form"
    history = body.get("history")
    if history is None:                            # fall back to stored sessions
        history = recent_sessions(limit=10)
    if not groq_service.is_configured():
        return _err("AI coach unavailable — GROQ_API_KEY is not configured.", 503)

    try:
        result = groq_service.generate_training_plan(user_goals, history)
    except Exception as exc:                       # noqa: BLE001
        return _err(f"AI coach error: {exc}", 502)
    return _ok(result)
