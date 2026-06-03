"""
evaluate_holdout.py  —  honest, leakage-free accuracy estimate.

train_squat/deadlift use a stratified RANDOM split, which is optimistic here:
windows overlap (stride 1), so near-duplicate neighbours land on both sides of
the split and inflate validation accuracy.

This script instead holds out the LAST 20% of each class's windows *contiguously*
(in file order). Overlapping windows of a class stay together, so the only
shared information is the single boundary window — a far more realistic estimate
of how the model does on unseen reps. It trains a throwaway model and reports;
it does NOT overwrite the saved models.

    python evaluate_holdout.py squat
    python evaluate_holdout.py deadlift
    python evaluate_holdout.py            # both
"""
from __future__ import annotations

import sys

import numpy as np

from lstm_common import (FEATURE_COLS, WINDOW, build_windows, load_enriched,
                         set_seeds, _build_model)


def _contiguous_holdout_idx(y: np.ndarray, frac: float = 0.2):
    """First (1-frac) of each class's windows -> train, last frac -> val."""
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]          # already in file order
        cut = int(len(idx) * (1 - frac))
        train_idx.extend(idx[:cut])
        val_idx.extend(idx[cut:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def evaluate(exercise: str, epochs: int = 60) -> None:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, accuracy_score
    from tensorflow.keras import callbacks

    set_seeds()
    df = load_enriched(exercise)
    X, y_str = build_windows(df)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)

    tr, va = _contiguous_holdout_idx(y_str)
    X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]

    nf = X.shape[2]
    scaler = StandardScaler().fit(X_tr.reshape(-1, nf))
    X_tr = scaler.transform(X_tr.reshape(-1, nf)).reshape(X_tr.shape).astype(np.float32)
    X_va = scaler.transform(X_va.reshape(-1, nf)).reshape(X_va.shape).astype(np.float32)

    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    model = _build_model(nf, len(classes))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=32, class_weight={i: w for i, w in enumerate(cw)},
              callbacks=[es], verbose=0)

    pred = model.predict(X_va, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_va, pred)
    print(f"\n=== {exercise}: leakage-free holdout accuracy = {acc:.3f} "
          f"(train {len(tr)} / val {len(va)} windows) ===")
    print(classification_report(y_va, pred, target_names=classes, zero_division=0))


if __name__ == "__main__":
    targets = sys.argv[1:] or ["squat", "deadlift"]
    for t in targets:
        evaluate(t)
