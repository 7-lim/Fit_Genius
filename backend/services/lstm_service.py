"""
lstm_service.py  —  Fit Genius, build-order step 4 (real-time inference).

Wraps the trained LSTMs and turns a stream of per-frame feature vectors into
live phase/form/rep feedback.

Key pieces (per CLAUDE.md):
  * Models load ONCE (lazy singletons), never per request.
  * SessionAnalyzer keeps a rolling 30-frame buffer per session and predicts the
    current phase + form from it.
  * Rep counting is a state machine here (UP->DOWN->UP = 1 rep), NOT in the LSTM.
  * A form error is only flagged when confidence > 0.75 for 3 consecutive frames,
    which kills flicker / false positives.

Class strings encode phase+form, e.g. 'down_deep' -> phase 'down', form 'deep'.
Plain 'up'/'down' -> form 'correct'.
"""
from __future__ import annotations

import pickle
from collections import Counter, deque
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
WINDOW = 30
CONF_THRESHOLD = 0.75          # min confidence to consider a form error real
ERROR_STREAK = 3               # consecutive frames before flagging the error
PHASE_DEBOUNCE = 2             # consecutive frames before a phase change counts

# Range-of-motion gate: a rep only counts if the primary joint angle actually
# swings through this many degrees. Rejects swaying / fidgeting / non-exercise
# movement that the closed-set LSTM would otherwise force into up/down phases.
MIN_REP_ROM = 40.0
# Which of the [knee, hip, spine] angles is the primary ROM joint per exercise.
PRIMARY_ANGLE_OFFSET = {"squat": 0, "deadlift": 1}

# Human-readable coaching labels for each form suffix.
FORM_LABELS = {
    "correct": "good form",
    "deep": "squatting too deep",
    "forward": "leaning too far forward",
    "low": "hips dropping too low / lost hip hinge",
    "roll": "rounding the lower back",
    "back": "hyperextending at lockout",
}


def _split_class(cls: str):
    """'down_deep' -> ('down', 'deep'); 'up' -> ('up', 'correct')."""
    phase = "up" if cls.startswith("up") else "down"
    form = cls.split("_", 1)[1] if "_" in cls else "correct"
    return phase, form


class _Model:
    """One exercise's model + scaler + label encoder, loaded once."""

    def __init__(self, exercise: str):
        import keras
        self.exercise = exercise
        meta = pickle.loads((MODELS_DIR / f"{exercise}_lstm.pkl").read_bytes())
        self.encoder = meta["label_encoder"]
        self.scaler = meta["scaler"]
        self.feature_cols = meta["feature_cols"]
        self.window = meta["window"]
        self.classes = meta["classes"]
        self.model = keras.models.load_model(MODELS_DIR / f"{exercise}_lstm.h5")

    def predict(self, window: np.ndarray):
        """window: (WINDOW, 135) raw features -> (class_str, confidence)."""
        nf = window.shape[1]
        scaled = self.scaler.transform(window.reshape(-1, nf)).reshape(1, self.window, nf)
        proba = self.model(scaled.astype("float32"), training=False).numpy()[0]
        idx = int(proba.argmax())
        return self.classes[idx], float(proba[idx])


_LOADED: dict[str, _Model] = {}


def get_model(exercise: str) -> _Model:
    """Lazy, cached model accessor (load once)."""
    if exercise not in _LOADED:
        if exercise not in ("squat", "deadlift"):
            raise ValueError(f"Unknown exercise: {exercise}")
        _LOADED[exercise] = _Model(exercise)
    return _LOADED[exercise]


def warm_up() -> None:
    """Eagerly load all models — call at app startup (CLAUDE.md)."""
    for ex in ("squat", "deadlift"):
        get_model(ex)


class SessionAnalyzer:
    """Stateful per-session analyzer: feed it one frame's features at a time."""

    def __init__(self, exercise: str):
        self.model = get_model(exercise)
        self.exercise = exercise
        self.buffer: deque = deque(maxlen=WINDOW)

        # rep-counting state machine
        self.rep_count = 0
        self._phase = None              # last *confirmed* phase
        self._descended = False         # saw up->down since last rep
        self._pending_phase = None
        self._pending_count = 0

        # range-of-motion gate (rejects sway / non-exercise as reps)
        self._primary_idx = -3 + PRIMARY_ANGLE_OFFSET[exercise]   # knee or hip
        self._cur_angle = 0.0
        self._rep_min = None            # min/max primary angle over current rep window
        self._rep_max = None

        # confidence gate for form errors
        self._streak_form = None
        self._streak_count = 0
        self.confirmed_errors: Counter = Counter()

        # running angle averages (for session summary / Groq agent later)
        self._angle_sum = np.zeros(3)
        self._angle_n = 0

    # -- internal helpers ---------------------------------------------------
    def _update_reps(self, phase: str) -> None:
        """Debounced UP->DOWN->UP rep counting."""
        if self._phase is None:                      # establish baseline immediately
            self._phase = self._pending_phase = phase
            self._pending_count = PHASE_DEBOUNCE
            return
        if phase == self._pending_phase:
            self._pending_count += 1
        else:
            self._pending_phase, self._pending_count = phase, 1

        if self._pending_count < PHASE_DEBOUNCE or phase == self._phase:
            return
        # confirmed phase change
        if self._phase == "up" and phase == "down":
            self._descended = True
        elif self._phase == "down" and phase == "up" and self._descended:
            # Only count if the joint actually moved through a real range of motion.
            rom = (self._rep_max - self._rep_min) if self._rep_min is not None else 0.0
            if rom >= MIN_REP_ROM:
                self.rep_count += 1
            self._descended = False
            self._rep_min = self._rep_max = self._cur_angle   # reset window for next rep
        self._phase = phase

    def _track_rom(self, features: np.ndarray) -> None:
        """Accumulate min/max of the primary joint angle over the current rep."""
        ang = float(features[self._primary_idx])
        self._cur_angle = ang
        self._rep_min = ang if self._rep_min is None else min(self._rep_min, ang)
        self._rep_max = ang if self._rep_max is None else max(self._rep_max, ang)

    def _update_error_gate(self, form: str, conf: float) -> bool:
        """Return True once an error has been confirmed for ERROR_STREAK frames."""
        if form != "correct" and conf > CONF_THRESHOLD:
            if form == self._streak_form:
                self._streak_count += 1
            else:
                self._streak_form, self._streak_count = form, 1
        else:
            self._streak_form, self._streak_count = None, 0

        if self._streak_count >= ERROR_STREAK:
            self.confirmed_errors[form] += 1
            return True
        return False

    # -- public API ---------------------------------------------------------
    def add_frame(self, features: np.ndarray) -> dict:
        """Add one 135-feature frame; returns the live analysis dict."""
        features = np.asarray(features, dtype=np.float32)
        self.buffer.append(features)
        knee, hip, spine = features[-3:]
        angles = {"knee": round(float(knee), 1),
                  "hip": round(float(hip), 1),
                  "spine": round(float(spine), 1)}

        if len(self.buffer) < WINDOW:
            return {
                "status": "warming_up",
                "frames_needed": WINDOW - len(self.buffer),
                "rep_count": self.rep_count,
                "angles": angles,
            }

        cls, conf = self.model.predict(np.stack(self.buffer))
        phase, form = _split_class(cls)
        self._track_rom(features)
        self._update_reps(phase)
        error_flagged = self._update_error_gate(form, conf)

        self._angle_sum += features[-3:]
        self._angle_n += 1

        return {
            "status": "analyzing",
            "phase": phase,
            "form_class": form,
            "form_label": FORM_LABELS.get(form, form),
            "confidence": round(conf, 3),
            "angles": angles,
            "rep_count": self.rep_count,
            "error_flagged": error_flagged,
            "feedback": self._live_feedback(form, error_flagged),
        }

    def _live_feedback(self, form: str, flagged: bool) -> str:
        if form == "correct":
            return "Looking good — keep it up!"
        if flagged:
            return f"Heads up: {FORM_LABELS.get(form, form)}."
        return ""

    def summary(self) -> dict:
        """End-of-session rollup (feeds session save + the Groq agent later)."""
        avg = (self._angle_sum / self._angle_n) if self._angle_n else np.zeros(3)
        return {
            "exercise": self.exercise,
            "reps": self.rep_count,
            "form_errors": dict(self.confirmed_errors),
            "avg_knee_angle": round(float(avg[0]), 1),
            "avg_hip_angle": round(float(avg[1]), 1),
            "avg_spine_angle": round(float(avg[2]), 1),
            "analyzed_frames": self._angle_n,
        }

    def reset(self) -> None:
        self.__init__(self.exercise)


# --------------------------------------------------------------------------- #
# Per-session analyzer registry (keeps API routes thin).
# Stateful across requests so rep counting / gating persist between frames.
# In-memory + single-process — fine for dev; swap for a store if scaling out.
# --------------------------------------------------------------------------- #
_ANALYZERS: dict[str, SessionAnalyzer] = {}


def get_analyzer(session_id: str, exercise: str) -> SessionAnalyzer:
    """Fetch/create the analyzer for a session; recreate if the exercise changed."""
    a = _ANALYZERS.get(session_id)
    if a is None or a.exercise != exercise:
        a = SessionAnalyzer(exercise)
        _ANALYZERS[session_id] = a
    return a


def get_summary(session_id: str):
    a = _ANALYZERS.get(session_id)
    return a.summary() if a else None


def reset_analyzer(session_id: str) -> None:
    _ANALYZERS.pop(session_id, None)


def analyze_video(exercise: str, video_path, target_fps: int = 6) -> dict:
    """Run a whole uploaded video through a fresh analyzer; return its summary.

    Frames are sampled to ~target_fps to match the real-time cadence the rep/ROM
    logic expects. Same SessionAnalyzer (incl. the ROM gate) as live analysis.
    """
    from services import mediapipe_service as mps          # lazy (heavy deps)
    analyzer = SessionAnalyzer(exercise)
    sampled = detected = 0
    for feats in mps.iter_video_features(video_path, target_fps=target_fps):
        sampled += 1
        if feats is not None:
            detected += 1
            analyzer.add_frame(feats)
    summary = analyzer.summary()
    summary["sampled_frames"] = sampled
    summary["detected_frames"] = detected
    return summary
