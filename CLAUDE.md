# CLAUDE.md — Fit Genius

## Project Overview

**Fit Genius** is an AI-powered fitness movement analysis web app.
It uses MediaPipe for real-time pose estimation, an LSTM model (TensorFlow/Keras)
to evaluate posture and detect form errors during squats and deadlifts, a Flask
REST API as the backend, and an Angular frontend for the UI. An AI Agent layer
powered by Groq (Llama 3, free tier) generates personalized training plans and
real-time coaching feedback based on the user's session data.

**Target exercises:** Squat, Deadlift
**Developer OS:** Windows
**Stack:** Angular + Flask + TensorFlow + MediaPipe + Groq API (Llama 3)

---

## Project Structure

```
fit-genius/
├── CLAUDE.md
├── README.md
│
├── backend/                        # Flask API
│   ├── app.py                      # Entry point, registers blueprints
│   ├── config.py                   # Config (env vars, paths, model paths)
│   ├── requirements.txt
│   ├── .env                        # GROQ_API_KEY, etc. (never commit)
│   │
│   ├── models/                     # Saved ML model files
│   │   ├── squat_lstm.h5
│   │   └── deadlift_lstm.h5
│   │
│   ├── training/                   # Offline training scripts (not served)
│   │   ├── data/
│   │   │   ├── squat/              # CSVs from Kaggle + Daniel repo
│   │   │   └── deadlift/           # CSVs from Daniel repo
│   │   ├── extract_landmarks.py    # MediaPipe → CSV pipeline
│   │   ├── train_squat.py
│   │   └── train_deadlift.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── pose.py                 # /api/pose/analyze  (frame analysis)
│   │   ├── session.py              # /api/session/save, /api/session/history
│   │   └── agent.py                # /api/agent/feedback, /api/agent/plan
│   │
│   ├── services/
│   │   ├── mediapipe_service.py    # Landmark extraction logic
│   │   ├── lstm_service.py         # Load models, run inference
│   │   └── groq_service.py         # Groq API wrapper + agent logic
│   │
│   └── db/
│       ├── database.py             # SQLAlchemy setup (SQLite for dev)
│       └── models.py               # Session, RepData ORM models
│
├── frontend/                       # Angular app (standalone, Angular 20)
│   ├── angular.json
│   ├── package.json
│   ├── proxy.conf.json             # /api → localhost:5000 (dev)
│   └── src/
│       └── app/
│           ├── app.config.ts        # providers (provideRouter, provideHttpClient)
│           ├── app.routes.ts        # route config (no NgModule)
│           ├── app.component.ts      # standalone root
│           ├── core/
│           │   ├── services/
│           │   │   ├── pose.service.ts       # Calls /api/pose/*
│           │   │   ├── session.service.ts    # Calls /api/session/*
│           │   │   └── agent.service.ts      # Calls /api/agent/*
│           │   └── models/
│           │       ├── session.model.ts
│           │       └── feedback.model.ts
│           └── pages/                # standalone components
│               ├── home/                     # Exercise picker
│               ├── workout/                  # Live webcam + feedback view
│               └── history/                  # Past sessions + AI plans
│
└── notebooks/                      # Exploratory work, kept separate
    ├── explore_squat_dataset.ipynb
    └── explore_deadlift_dataset.ipynb
```

---

## Datasets

| Exercise | Source | Format | Notes |
|----------|--------|--------|-------|
| Squat | [Kaggle — Thashmila Dewmini](https://www.kaggle.com/datasets/thashmiladewmini/squat-exercise-pose-dataset) | CSV (landmarks pre-extracted) | Multiple form classes |
| Deadlift | [DanielGuarnizo GitHub repo](https://github.com/DanielGuarnizo/Pose-Estimation-for-Fitness-Exercise-Analysis) | CSV (landmarks) + raw videos | UP/DOWN phases + form error classes |

Place datasets under `backend/training/data/squat/` and `backend/training/data/deadlift/`.
Raw videos for deadlift go in a `videos/` subfolder if re-extraction is needed.

---

## ML Pipeline

### Step 1 — Landmark Extraction (`extract_landmarks.py`)
- Input: video file or pre-extracted CSV
- Pose estimation: MediaPipe **Tasks API** `PoseLandmarker` with the
  `pose_landmarker_full.task` bundle (the installed mediapipe 0.10.35 / Py 3.13
  no longer ships the legacy `mediapipe.solutions.pose`)
- Extract 33 landmarks × (x, y, z, visibility) = 132 features per frame
- Also compute key joint angles: knee, hip, spine — these are the most
  informative features for form detection
- Output: CSV with columns `[label, lm_0_x, lm_0_y, ..., angle_knee, angle_hip, angle_spine]`

### Step 2 — Sequence Building
- Group frames into sliding windows of 30 frames (1 rep ≈ 30 frames at 30fps)
- Shape per sample: `(30, 132+angles)`
- Labels: phase (UP/DOWN) + form class (CORRECT, KNEE_CAVE, BACK_ROUND, etc.)

### Step 3 — LSTM Model (`train_squat.py`, `train_deadlift.py`)
```python
# Target architecture
Input → LSTM(128) → Dropout(0.3) → LSTM(64) → Dense(64, relu) → Dense(num_classes, softmax)
```
- Use `tf.keras` (TensorFlow 2.x)
- Save as `.h5` to `backend/models/`
- Log accuracy, confusion matrix per class

### Step 4 — Real-time Inference
- Frontend sends base64-encoded webcam frames to `/api/pose/analyze`
- Backend decodes frame → MediaPipe extracts landmarks → LSTM predicts phase + form
- Returns JSON: `{ phase, form_class, confidence, angles, feedback_text }`

---

## API Endpoints

```
POST /api/pose/analyze
  Body: { frame: "<base64 jpg>", exercise: "squat"|"deadlift" }
  Returns: { phase, form_class, confidence, angles, rep_count, feedback }

POST /api/session/save
  Body: { exercise, reps, form_errors: [...], duration_sec }
  Returns: { session_id }

GET  /api/session/history
  Returns: [ { session_id, date, exercise, reps, top_errors } ]

POST /api/agent/feedback
  Body: { session_id }
  Returns: { summary, corrections, tips }   ← Groq Llama 3 generated

POST /api/agent/plan
  Body: { user_goals, history: [...last N sessions] }
  Returns: { weekly_plan, focus_areas, progression_notes }  ← Groq Llama 3
```

---

## AI Agent — Groq Integration

**Model:** `llama-3.1-8b-instant` via Groq API (free tier, very fast). NOTE:
the original `llama3-8b-8192` has been decommissioned on Groq; the model is
configurable via the `GROQ_MODEL` env var.
**Library:** `groq` Python SDK

### groq_service.py pattern
```python
from groq import Groq

client = Groq(api_key=os.environ[""])

def generate_session_feedback(session_data: dict) -> str:
    prompt = build_feedback_prompt(session_data)  # structured prompt with real data
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return response.choices[0].message.content
```

### System prompt style
```
You are FitGenius Coach, an expert personal trainer and sports physiotherapist.
You receive structured JSON data about a user's workout session including exercise type,
rep count, detected form errors, and joint angles. You respond with:
1. A brief session summary (2-3 sentences)
2. The top 1-2 form corrections with clear anatomical explanations
3. One actionable drill to fix the main issue
Be direct, encouraging, and specific. Never generic.
```

### Agent inputs (always structured, never raw)
```python
def build_feedback_prompt(session: dict) -> str:
    return f"""
Exercise: {session['exercise']}
Reps completed: {session['reps']}
Session duration: {session['duration_sec']}s
Form errors detected:
{json.dumps(session['form_errors'], indent=2)}
Average joint angles:
- Knee: {session['avg_knee_angle']:.1f}°
- Hip: {session['avg_hip_angle']:.1f}°
- Spine deviation: {session['avg_spine_angle']:.1f}°

Provide post-session coaching feedback.
"""
```

---

## Environment Setup (Windows)

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install flask flask-cors sqlalchemy mediapipe tensorflow groq python-dotenv opencv-python

# Frontend
cd frontend
npm install
ng serve --proxy-config proxy.conf.json   # proxy /api → localhost:5000
```

### .env (backend root, never commit)
```
GROQ_API_KEY=your_key_here
FLASK_ENV=development
MODEL_DIR=models/
DATABASE_URL=sqlite:///fitgenius.db
```

### proxy.conf.json (Angular, for dev)
```json
{
  "/api": {
    "target": "http://localhost:5000",
    "secure": false
  }
}
```

---

## Development Conventions

### Python (Flask backend)
- Flask blueprints — one per API domain (`pose`, `session`, `agent`)
- Services layer handles all business logic; routes are thin
- Never import TensorFlow or MediaPipe in route files — always go through services
- Return consistent JSON: `{ data: ..., error: null }` or `{ data: null, error: "message" }`
- All model loading happens once at app startup (not per request)

### TypeScript (Angular frontend)
- All API calls go through service classes in `core/services/`
- Components never call `fetch` or `HttpClient` directly
- Use `async/await` with RxJS `firstValueFrom()` for HTTP calls in components
- Webcam frames captured with `<video>` + `<canvas>` approach (no extra libs needed)
- Polling interval for live analysis: 200ms (5 fps is enough for form detection)

### ML / Training
- Training scripts are standalone — they do not import anything from `api/`
- Always set `random_seed = 42` for reproducibility
- Save both the model (`.h5`) and the label encoder (`.pkl`) together
- Validate on a held-out person's data if possible (not just random split)

---

## Key Implementation Notes

1. **Frame rate for analysis:** Send 1 frame every 200ms from frontend (not every frame).
   Real-time feel without hammering the API.

2. **Rep counting logic:** Track phase transitions UP→DOWN→UP = 1 rep.
   Implement in `lstm_service.py` with a simple state machine, not in the LSTM itself.

3. **Confidence threshold:** Only flag a form error if confidence > 0.75 for 3 consecutive
   frames — avoids flickering false positives.

4. **MediaPipe model complexity:** Use `model_complexity=1` (balanced).
   `complexity=2` is too slow for real-time on CPU-only Windows machines.

5. **Groq rate limits (free tier):** ~30 req/min. Agent calls happen post-session only,
   never during live analysis. This is by design.

6. **CORS:** Flask-CORS enabled for `http://localhost:4200` in development only.

7. **No GPU assumed:** TensorFlow CPU-only. The LSTM is small enough that inference
   on CPU is under 10ms per frame.

---

## Resume Description (final)

```
Fit Genius — AI Fitness Coach | TensorFlow, MediaPipe, Flask, Angular, Groq
- Extracted 132-dimensional pose landmarks per frame using MediaPipe for squat
  and deadlift analysis from two open datasets (Kaggle + GitHub).
- Trained LSTM model (TensorFlow/Keras) on 30-frame sliding windows to classify
  exercise phase (UP/DOWN) and detect form errors (knee cave, back rounding, etc.).
- Built REST API (Flask) serving real-time inference at 5fps from webcam stream.
- Integrated Groq (Llama 3) AI agent to generate post-session coaching feedback
  and personalized weekly training plans based on detected form error history.
- Full-stack web interface with Angular frontend and live visual pose overlay.
```

---

## Build Order (recommended)

1. `notebooks/` — explore both datasets, understand label distributions
2. `training/extract_landmarks.py` — build unified CSV from both sources
3. `training/train_squat.py` + `train_deadlift.py` — train and save models
4. `services/mediapipe_service.py` + `services/lstm_service.py` — inference pipeline
5. `api/pose.py` — `/api/pose/analyze` endpoint, test with Postman
6. Angular `workout/` page — webcam capture + polling + overlay
7. `db/` + `api/session.py` — session persistence
8. `services/groq_service.py` + `api/agent.py` — AI agent layer
9. Angular `history/` page — past sessions + AI plan display
10. Polish, error handling, README
