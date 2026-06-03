# 🏋️ Fit Genius — AI Fitness Form Coach

Real-time **squat** and **deadlift** form analysis in the browser. MediaPipe
estimates your pose, an LSTM classifies the exercise phase and form errors, and a
Groq-hosted Llama model turns each session into personalized coaching feedback and
weekly training plans.

**Stack:** Angular 20 · Flask · TensorFlow/Keras (LSTM) · MediaPipe · Groq (Llama 3.1) · SQLite

---

## Features

- 🎥 **Live analysis** — webcam streamed at 5 fps to the backend; live pose overlay, phase (up/down), form errors, rep counter, and confidence.
- ⬆️ **Video upload** — analyze a pre-recorded clip instead of live camera.
- 🔢 **Robust rep counting** — a range-of-motion gate rejects swaying/fidgeting so only real reps count.
- 🧠 **Form-error detection** — squat: too deep, forward lean; deadlift: back rounding, hips too low, lockout hyperextension.
- 🤖 **AI coach (Groq)** — post-session feedback (summary + corrections + a drill) and a personalized weekly plan.
- 📊 **History** — past sessions persisted to SQLite, viewable with top issues.

---

## Architecture

```
Browser (Angular)
  └─ webcam frame (base64, 5fps) ─▶ POST /api/pose/analyze
                                       │
Flask API ──▶ mediapipe_service  → 33 landmarks + knee/hip/spine angles (135 features)
          ──▶ lstm_service       → rolling 30-frame window → LSTM → phase/form
                                    rep state-machine + ROM gate + confidence gate
          ──▶ db (SQLite)        → save / history
          ──▶ groq_service       → LLM feedback & weekly plan
```

The angle/feature code is shared between training and inference, so the live
feature vector is identical to what the models were trained on.

---

## Prerequisites

- **Python 3.13** and **Node.js 20+** (Angular CLI 20)
- A free **Groq API key** for the AI coach — https://console.groq.com

---

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Add your Groq key (copy the template):

```bash
copy .env.example .env           # Windows  (cp on macOS/Linux)
# then edit .env and set GROQ_API_KEY=gsk_...
```

The MediaPipe pose model bundle is required at `backend/models/pose_landmarker_full.task`.
If it is missing, download it:

```bash
curl -L -o models/pose_landmarker_full.task ^
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

> The trained LSTM models (`squat_lstm.h5`, `deadlift_lstm.h5` + their `.pkl`) ship
> in `backend/models/`. To retrain, see [Training](#training-optional).

### 2. Frontend

```bash
cd frontend
npm install
```

---

## Running

Two terminals:

```bash
# 1) backend  →  http://localhost:5000
cd backend && python app.py

# 2) frontend →  http://localhost:4200   (proxies /api to the backend)
cd frontend && ng serve
```

Open **http://localhost:4200**, pick an exercise, allow camera access (or upload a
video), do your reps, hit **Finish session**, then **Get AI coaching feedback**.
Past sessions and AI plans live on the **History** page.

---

## API

| Method & path | Body | Returns |
|---|---|---|
| `POST /api/pose/analyze` | `{ frame, exercise, session_id }` | `{ phase, form_class, confidence, angles, rep_count, feedback, landmarks }` |
| `POST /api/pose/analyze-video` | multipart: `video`, `exercise` | session summary (`reps`, `form_errors`, avg angles, frame counts) |
| `POST /api/pose/reset` | `{ session_id }` | `{ summary }` |
| `POST /api/session/save` | `{ exercise, reps, form_errors, duration_sec, avg_* }` | `{ session_id }` |
| `GET  /api/session/history` | — | `[ { session_id, date, exercise, reps, top_errors, ... } ]` |
| `POST /api/agent/feedback` | `{ session_id }` | `{ summary, corrections, tips }` |
| `POST /api/agent/plan` | `{ user_goals, history? }` | `{ weekly_plan, focus_areas, progression_notes }` |

All responses use the envelope `{ "data": ..., "error": null }`.

---

## ML pipeline

- **Data:** pre-extracted MediaPipe landmark CSVs — squat (`coords_SQ_C.csv`) and
  deadlift (`coords_DL_C.csv`). Each frame = 33 landmarks × (x, y, z, visibility) = 132
  features, plus computed knee/hip/spine angles → 135 features.
- **Model:** 30-frame sliding windows → `LSTM(128) → Dropout(0.3) → LSTM(64) → Dense(64) → softmax`.
- **Honest accuracy** (leakage-free per-class holdout, not the optimistic random-split number):
  **squat ≈ 0.89**, **deadlift ≈ 1.00**. Run `python training/evaluate_holdout.py` to reproduce.

### Training (optional)

```bash
cd backend/training
python extract_landmarks.py          # build *_enriched.csv (landmarks + angles)
python train_squat.py                # -> ../models/squat_lstm.h5 + .pkl
python train_deadlift.py             # -> ../models/deadlift_lstm.h5 + .pkl
python evaluate_holdout.py           # honest accuracy estimate
```

Exploratory notebooks are in `notebooks/`.

---

## Implementation notes

- **MediaPipe Tasks API:** the installed `mediapipe 0.10.x` (Python 3.13) ships only
  the Tasks API, so this uses `PoseLandmarker` + the `pose_landmarker_full.task` bundle
  (not the legacy `mediapipe.solutions.pose`).
- **Groq model:** `llama-3.1-8b-instant` (the older `llama3-8b-8192` is decommissioned);
  override with the `GROQ_MODEL` env var.
- **Rep counting** is a state machine (UP→DOWN→UP) with a **range-of-motion gate** so
  non-exercise movement isn't counted; form errors are flagged only after 3 consecutive
  confident frames.
- **CPU-only** TensorFlow; inference is a few ms per frame.

---

## Troubleshooting

- **`npm install` fails with ENOSPC / "No space left"** — point npm's cache and temp to a
  drive with space, e.g. `npm_config_cache=D:/npm-cache TMP=D:/npm-tmp TEMP=D:/npm-tmp npm install`.
- **"No person detected"** — make sure your full body is in frame and well lit.
- **AI feedback returns 503** — `GROQ_API_KEY` is missing from `backend/.env`.

---

## Project structure

```
backend/   Flask API (api/), services/ (mediapipe, lstm, groq), db/, training/, models/
frontend/  Angular 20 standalone app (pages/, core/services/, core/models/, shared/)
notebooks/ dataset exploration
```
