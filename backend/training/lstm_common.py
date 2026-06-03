"""
lstm_common.py  —  Fit Genius, shared training logic for build-order step 3.

train_squat.py and train_deadlift.py are thin wrappers around train_exercise()
here, so the windowing / model / save logic lives in exactly one place.

Pipeline
--------
enriched CSV  ->  30-frame sliding windows (label = current/last frame)
              ->  stratified 80/20 split  ->  feature standardization
              ->  LSTM(128) -> Dropout -> LSTM(64) -> Dense(64) -> softmax
              ->  saved <exercise>_lstm.h5  +  <exercise>_lstm.pkl (encoder+scaler+meta)
              ->  classification report + confusion-matrix PNG

Windowing rationale
-------------------
The coords frames are real reps (up/down alternation) but most same-class runs
are < 30 frames, so a window deliberately spans phase transitions and is labelled
by its LAST frame — the "current state". That matches live inference, where a
rolling 30-frame buffer predicts the current phase+form.

The form-error classes are clustered by position in the file (each condition was
recorded separately), so a temporal split would drop whole classes from training.
We therefore use a STRATIFIED RANDOM split. Because windows overlap (stride 1),
adjacent near-identical windows can land on both sides of the split, so reported
validation accuracy is optimistic — a known, accepted trade-off of using the
pre-extracted CSVs as-is (no rep/clip boundaries to split on).
"""
from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

HERE = Path(__file__).resolve().parent          # backend/training
PROCESSED = HERE / "data" / "processed"
MODELS = HERE.parent / "models"                 # backend/models
REPORTS = HERE / "reports"

WINDOW = 30                                      # frames per sample (~1 rep @ 30fps)
STRIDE = 1                                       # slide step (1 = max samples)

LANDMARK_COLS = [f"{a}{i}" for i in range(1, 34) for a in ("x", "y", "z", "v")]
ANGLE_COLS = ["angle_knee", "angle_hip", "angle_spine"]
FEATURE_COLS = LANDMARK_COLS + ANGLE_COLS       # 135 features


def set_seeds() -> None:
    """Make a run reproducible (CLAUDE.md: random_seed = 42)."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf
    tf.random.set_seed(SEED)


def load_enriched(exercise: str) -> pd.DataFrame:
    path = PROCESSED / f"{exercise}_enriched.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run extract_landmarks.py first."
        )
    return pd.read_csv(path)


def build_windows(df: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE):
    """Sliding windows over frame order. Label = class of the window's last frame.

    Returns X (n, window, 135) float32 and y (n,) of class strings.
    """
    feats = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = df["class"].to_numpy()
    xs, ys = [], []
    for end in range(window, len(df) + 1, stride):
        xs.append(feats[end - window:end])
        ys.append(labels[end - 1])              # current state = last frame
    return np.asarray(xs, dtype=np.float32), np.asarray(ys)


def _build_model(n_features: int, n_classes: int):
    """LSTM(128) -> Dropout(0.3) -> LSTM(64) -> Dense(64) -> softmax (per CLAUDE.md)."""
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input((WINDOW, n_features)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _save_confusion(exercise, y_true, y_pred, classes) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 3, 1.0 * len(classes) + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(title=f"{exercise} LSTM — confusion matrix (validation)",
           xlabel="predicted", ylabel="true")
    plt.tight_layout()
    out = REPORTS / f"{exercise}_confusion.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    return out


def train_exercise(exercise: str, epochs: int = 60, batch_size: int = 32) -> dict:
    """Train, evaluate, and save the LSTM for one exercise. Returns a summary dict."""
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, accuracy_score
    from tensorflow.keras import callbacks

    set_seeds()
    print(f"\n=== Training {exercise} ===")
    df = load_enriched(exercise)
    X, y_str = build_windows(df)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)
    print(f"  windows: {X.shape[0]}  shape/sample: {X.shape[1:]}  classes: {classes}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Standardize features (fit on TRAIN only) so 0-180 angles don't dominate
    # the 0-1 landmark coords. Reshape to 2D for the scaler, then back to 3D.
    nf = X.shape[2]
    scaler = StandardScaler().fit(X_tr.reshape(-1, nf))
    X_tr = scaler.transform(X_tr.reshape(-1, nf)).reshape(X_tr.shape).astype(np.float32)
    X_te = scaler.transform(X_te.reshape(-1, nf)).reshape(X_te.shape).astype(np.float32)

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: w for i, w in enumerate(cw)}

    model = _build_model(nf, len(classes))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=epochs, batch_size=batch_size,
              class_weight=class_weight, callbacks=[es], verbose=2)

    y_pred = model.predict(X_te, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n  validation accuracy: {acc:.3f}\n")
    print(classification_report(y_te, y_pred, target_names=classes, zero_division=0))

    cm_path = _save_confusion(exercise, y_te, y_pred, classes)

    MODELS.mkdir(parents=True, exist_ok=True)
    h5_path = MODELS / f"{exercise}_lstm.h5"
    model.save(h5_path)
    pkl_path = MODELS / f"{exercise}_lstm.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "label_encoder": le,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "window": WINDOW,
            "classes": classes,
        }, f)

    print(f"  saved model   -> {h5_path}")
    print(f"  saved encoder -> {pkl_path}")
    print(f"  saved cm      -> {cm_path}")
    return {"exercise": exercise, "val_accuracy": acc,
            "n_windows": int(X.shape[0]), "classes": classes}
