"""
groq_service.py  —  Fit Genius AI coaching agent (build-order step 8).

Wraps the Groq LLM. Two capabilities:
  * generate_session_feedback(session) -> {summary, corrections, tips}
  * generate_training_plan(goals, history) -> {weekly_plan, focus_areas, progression_notes}

Inputs are always STRUCTURED (real session data formatted into the prompt),
never raw. Output uses Groq's JSON mode so the API contract is reliable.

Model: configurable via GROQ_MODEL. CLAUDE.md specifies `llama3-8b-8192`, but
that has been decommissioned on Groq; the default below is a current free-tier
Llama model. The client and key are read lazily so importing this module never
fails when GROQ_API_KEY is absent.
"""
from __future__ import annotations

import json
import os

from config import Config

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

COACH_SYSTEM_PROMPT = """\
You are FitGenius Coach, an expert personal trainer and sports physiotherapist.
You receive structured JSON data about a user's workout session including exercise
type, rep count, detected form errors, and joint angles. You respond with:
1. A brief session summary (2-3 sentences)
2. The top 1-2 form corrections with clear anatomical explanations
3. One actionable drill to fix the main issue
Be direct, encouraging, and specific. Never generic.

Respond ONLY with a JSON object of this exact shape:
{"summary": "<2-3 sentence summary>",
 "corrections": ["<correction 1>", "<correction 2>"],
 "tips": ["<one actionable drill or cue>"]}"""

PLAN_SYSTEM_PROMPT = """\
You are FitGenius Coach, an expert strength coach. Given a user's goals and their
recent squat/deadlift session history (reps and recurring form errors), design a
focused weekly plan that addresses their weaknesses and progresses safely.
Be specific and practical.

Respond ONLY with a JSON object of this exact shape:
{"weekly_plan": [{"day": "<e.g. Monday>", "focus": "<focus>", "work": "<exercises/sets/reps>"}],
 "focus_areas": ["<area 1>", "<area 2>"],
 "progression_notes": "<how to progress week to week>"}"""

_client = None


def is_configured() -> bool:
    """True if a GROQ_API_KEY is available."""
    return bool(Config.GROQ_API_KEY)


def _get_client():
    global _client
    if _client is None:
        if not Config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in the environment/.env")
        from groq import Groq
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


def _chat_json(system: str, user: str, max_tokens: int = 700,
               temperature: float = 0.4) -> dict:
    resp = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _fmt_angle(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"


def build_feedback_prompt(session: dict) -> str:
    return f"""\
Exercise: {session.get('exercise')}
Reps completed: {session.get('reps', 0)}
Session duration: {session.get('duration_sec', 0)}s
Form errors detected (name: times flagged):
{json.dumps(session.get('form_errors', {}), indent=2)}
Average joint angles:
- Knee: {_fmt_angle(session.get('avg_knee_angle'))} deg
- Hip: {_fmt_angle(session.get('avg_hip_angle'))} deg
- Spine deviation: {_fmt_angle(session.get('avg_spine_angle'))} deg

Provide post-session coaching feedback."""


def build_plan_prompt(user_goals: str, history: list[dict]) -> str:
    rows = [
        {
            "exercise": h.get("exercise"),
            "reps": h.get("reps"),
            "top_errors": h.get("top_errors", []),
        }
        for h in (history or [])
    ]
    return f"""\
User goals: {user_goals}
Recent sessions (most recent first):
{json.dumps(rows, indent=2) if rows else "No sessions recorded yet."}

Design a personalized weekly training plan."""


def generate_session_feedback(session: dict) -> dict:
    return _chat_json(COACH_SYSTEM_PROMPT, build_feedback_prompt(session), max_tokens=600)


def generate_training_plan(user_goals: str, history: list[dict]) -> dict:
    return _chat_json(PLAN_SYSTEM_PROMPT, build_plan_prompt(user_goals, history),
                      max_tokens=900)
