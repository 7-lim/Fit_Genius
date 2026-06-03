"""
pose.py  —  /api/pose/* blueprint (build-order step 5).

Thin routes over the services layer. Live form analysis is stateful per session
(rep counting + the confidence gate need history), so the analyzer is keyed by
session_id and persists between requests.

Responses use the project's consistent envelope:
    success -> { "data": ..., "error": null }
    failure -> { "data": null, "error": "message" }   (with a 4xx/5xx status)
"""
import os
import tempfile

from flask import Blueprint, current_app, jsonify, request

from services import mediapipe_service as mps
from services.lstm_service import (analyze_video, get_analyzer, get_summary,
                                   reset_analyzer)

pose_bp = Blueprint("pose", __name__, url_prefix="/api/pose")

VALID_EXERCISES = {"squat", "deadlift"}


def _ok(data):
    return jsonify({"data": data, "error": None})


def _err(message, status=400):
    return jsonify({"data": None, "error": message}), status


@pose_bp.post("/analyze")
def analyze():
    """One webcam frame in -> live phase/form/rep analysis out."""
    body = request.get_json(silent=True) or {}
    frame = body.get("frame")
    exercise = body.get("exercise")
    session_id = body.get("session_id", "default")

    if not frame:
        return _err("Missing 'frame' (base64 image).")
    if exercise not in VALID_EXERCISES:
        return _err(f"'exercise' must be one of {sorted(VALID_EXERCISES)}.")

    try:
        features, landmarks = mps.frame_to_features(frame)
    except ValueError as exc:
        return _err(f"Could not decode frame: {exc}")

    analyzer = get_analyzer(session_id, exercise)

    if features is None:
        return _ok({
            "status": "no_pose",
            "rep_count": analyzer.rep_count,
            "feedback": "No person detected — make sure your full body is in frame.",
        })

    result = analyzer.add_frame(features)
    result["landmarks"] = landmarks          # 33 points for the frontend overlay
    return _ok(result)


@pose_bp.post("/analyze-video")
def analyze_video_route():
    """Analyze a whole uploaded video (multipart: 'video' file + 'exercise')."""
    exercise = request.form.get("exercise")
    if exercise not in VALID_EXERCISES:
        return _err(f"'exercise' must be one of {sorted(VALID_EXERCISES)}.")
    file = request.files.get("video")
    if not file or not file.filename:
        return _err("Missing 'video' file upload.")

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=current_app.config["UPLOAD_TMP"])
    os.close(fd)
    try:
        file.save(path)
        summary = analyze_video(exercise, path)
    except Exception as exc:                       # noqa: BLE001 (report any decode/processing failure)
        return _err(f"Could not process video: {exc}", 500)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if summary["detected_frames"] == 0:
        summary["warning"] = (
            "No pose detected — make sure the full body is visible in the video."
        )
    return _ok(summary)


@pose_bp.post("/reset")
def reset():
    """End/clear a live session; returns its final summary (for session save)."""
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "default")
    summary = get_summary(session_id)
    reset_analyzer(session_id)
    return _ok({"status": "reset", "session_id": session_id, "summary": summary})
