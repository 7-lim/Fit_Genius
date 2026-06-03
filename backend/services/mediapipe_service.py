"""
mediapipe_service.py  —  Fit Genius, build-order step 4 (landmark extraction).

Decodes a webcam frame and turns it into the exact 135-feature vector the LSTMs
were trained on (132 MediaPipe landmarks + knee/hip/spine angles). Feature parity
is guaranteed by reusing `add_angle_columns` from the training pipeline — if the
angle math ever drifts, both training and inference move together.

MediaPipe note: the installed wheel (0.10.35, Python 3.13) ships only the modern
**Tasks API**, not the legacy `mp.solutions.pose`. We therefore use
`PoseLandmarker` with the `pose_landmarker_full` bundle ("full" == the balanced
model, equivalent to the old model_complexity=1). It runs in IMAGE mode because
frames arrive as independent HTTP requests; temporal modelling is handled by the
LSTM's 30-frame window, not by MediaPipe tracking.

The landmarker is a lazy singleton, so importing this module is cheap and never
fails if MediaPipe isn't installed.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the SINGLE source of truth for feature engineering (keeps inference
# features byte-for-byte identical to what the models were trained on).
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
from training.extract_landmarks import (LANDMARK_COLS, ANGLE_COLS,  # noqa: E402
                                        add_angle_columns)

FEATURE_COLS = LANDMARK_COLS + ANGLE_COLS          # 135, same order as training
POSE_MODEL_PATH = _BACKEND / "models" / "pose_landmarker_full.task"

_landmarker = None          # lazy PoseLandmarker singleton


def _get_landmarker():
    """Create (once) and return the MediaPipe Tasks PoseLandmarker."""
    global _landmarker
    if _landmarker is None:
        if not POSE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing pose model bundle: {POSE_MODEL_PATH}. Download "
                "pose_landmarker_full.task from the MediaPipe model garden."
            )
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = vision.PoseLandmarker.create_from_options(opts)
    return _landmarker


def decode_base64_frame(b64: str) -> np.ndarray:
    """'data:image/jpeg;base64,...' or bare base64 -> BGR image (H, W, 3)."""
    import cv2
    if "," in b64[:32]:                            # strip data-URL prefix if present
        b64 = b64.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode frame (not a valid base64 image)")
    return img


def extract_features(frame_bgr: np.ndarray):
    """Run MediaPipe on one BGR frame.

    Returns (features, landmarks):
      features  : np.ndarray (135,) float32 — or None if no pose detected
      landmarks : list of {x, y, z, visibility} for the 33 points (overlay) — or None
    """
    import cv2
    import mediapipe as mp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _get_landmarker().detect(mp_image)
    if not res.pose_landmarks:                     # empty list -> no pose
        return None, None

    flat: list[float] = []
    overlay: list[dict] = []
    for lm in res.pose_landmarks[0]:               # first (only) pose, 33 landmarks
        vis = float(lm.visibility) if lm.visibility is not None else 0.0
        flat += [lm.x, lm.y, lm.z, vis]
        overlay.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": vis})

    row = pd.DataFrame([flat], columns=LANDMARK_COLS)
    feats = add_angle_columns(row)[FEATURE_COLS].to_numpy(dtype=np.float32)[0]
    return feats, overlay


def frame_to_features(b64: str):
    """Convenience: base64 frame -> (features (135,), landmarks) or (None, None)."""
    return extract_features(decode_base64_frame(b64))


def iter_video_features(video_path, target_fps: int = 6):
    """Yield the 135-feature vector (or None) for frames of a video, sampled to
    ~target_fps so the cadence matches the live 5fps path."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / target_fps))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                feats, _ = extract_features(frame)
                yield feats
            idx += 1
    finally:
        cap.release()
