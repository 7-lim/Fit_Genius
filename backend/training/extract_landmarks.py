"""
extract_landmarks.py  —  Fit Genius, build-order step 2.

Turns pose data into the model-ready, per-frame feature schema:

    [class, x1, y1, z1, v1, ..., x33, y33, z33, v33, angle_knee, angle_hip, angle_spine]

Two input paths (per CLAUDE.md "input: video file OR pre-extracted CSV"):

  1. CSV enrichment (the path Fit Genius uses) — read the pre-extracted
     coords CSVs and append the three joint angles. No MediaPipe needed.

  2. Video extraction — run MediaPipe over a clip to produce the 132 landmark
     features, then append the same angles. MediaPipe/OpenCV are imported
     lazily so path 1 works even before they are installed.

The three angles (knee, hip, spine) are the most informative features for form
detection and feed both training and live inference (avg_knee/hip/spine angle).

Usage
-----
    # enrich both coords CSVs -> backend/training/data/processed/*_enriched.csv
    python extract_landmarks.py

    # extract a single labelled video
    python extract_landmarks.py --video clip.mp4 --label correct --out out.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent                      # backend/training
DATA = HERE / "data"
COORDS = DATA / "CSV_files"
PROCESSED = DATA / "processed"

# Pre-extracted coords CSVs we actually train on (squat + deadlift only;
# bench press and the synthetic augmented set are intentionally excluded).
SOURCES = {
    "squat": COORDS / "coords_SQ_C.csv",
    "deadlift": COORDS / "coords_DL_C.csv",
}

# --------------------------------------------------------------------------- #
# MediaPipe Pose landmark indices (0-indexed, BlazePose 33-point model).
# CSV columns are 1-indexed: landmark k -> x{k+1}, y{k+1}, z{k+1}, v{k+1}.
# --------------------------------------------------------------------------- #
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Feature schema (single source of truth, reused by training + inference).
LANDMARK_COLS = [f"{a}{i}" for i in range(1, 34) for a in ("x", "y", "z", "v")]
ANGLE_COLS = ["angle_knee", "angle_hip", "angle_spine"]


# --------------------------------------------------------------------------- #
# Angle geometry (2D, on normalized x/y — z is noisier and not needed here)
# --------------------------------------------------------------------------- #
def _xy(df: pd.DataFrame, k: int) -> np.ndarray:
    """(N, 2) array of the x,y for MediaPipe landmark index k."""
    return df[[f"x{k + 1}", f"y{k + 1}"]].to_numpy(dtype=float)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle (degrees) at vertex b of the path a-b-c, vectorized."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    cos = np.clip((ba * bc).sum(axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _vertical_deviation(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Angle (degrees) of the vector bottom->top away from straight up.

    0deg = perfectly upright torso; larger = more lean/rounding. Image y grows
    downward, so 'up' is (0, -1)."""
    v = top - bottom
    cos = np.clip((v * np.array([0.0, -1.0])).sum(axis=1)
                  / (np.linalg.norm(v, axis=1) + 1e-8), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def add_angle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append angle_knee, angle_hip, angle_spine to a landmark DataFrame.

    - knee : mean of L/R hip-knee-ankle angles
    - hip  : mean of L/R shoulder-hip-knee angles
    - spine: torso (mid-hip -> mid-shoulder) deviation from vertical
    """
    l_sh, r_sh = _xy(df, L_SHOULDER), _xy(df, R_SHOULDER)
    l_hip, r_hip = _xy(df, L_HIP), _xy(df, R_HIP)
    l_knee, r_knee = _xy(df, L_KNEE), _xy(df, R_KNEE)
    l_ank, r_ank = _xy(df, L_ANKLE), _xy(df, R_ANKLE)

    knee = (_joint_angle(l_hip, l_knee, l_ank)
            + _joint_angle(r_hip, r_knee, r_ank)) / 2.0
    hip = (_joint_angle(l_sh, l_hip, l_knee)
           + _joint_angle(r_sh, r_hip, r_knee)) / 2.0
    spine = _vertical_deviation((l_sh + r_sh) / 2.0, (l_hip + r_hip) / 2.0)

    out = df.copy()
    out["angle_knee"] = knee
    out["angle_hip"] = hip
    out["angle_spine"] = spine
    return out


# --------------------------------------------------------------------------- #
# Path 1 — enrich a pre-extracted coords CSV
# --------------------------------------------------------------------------- #
def enrich_coords_csv(in_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    n_feat = df.shape[1] - 1  # minus the `class` column
    if n_feat != 132:
        raise ValueError(
            f"{in_path.name}: expected 132 landmark features, found {n_feat}"
        )
    enriched = add_angle_columns(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    return enriched


# --------------------------------------------------------------------------- #
# Path 2 — extract landmarks from a video with MediaPipe (lazy import)
# --------------------------------------------------------------------------- #
def extract_landmarks_from_video(video_path: Path, label: str,
                                 pose_model: Path | None = None) -> pd.DataFrame:
    """Run MediaPipe Pose over a clip -> DataFrame [class, x1..v33, angles].

    Uses the Tasks-API PoseLandmarker with the 'full' bundle (balanced, the
    equivalent of the legacy model_complexity=1). The 0.10.35 wheel no longer
    ships mp.solutions.pose. Needs pose_landmarker_full.task (defaults to
    backend/models/)."""
    try:
        import cv2  # noqa: WPS433  (lazy: only needed for the video path)
        import mediapipe as mp  # noqa: WPS433
        from mediapipe.tasks.python import vision  # noqa: WPS433
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Video extraction needs mediapipe + opencv-python. "
            "Install them or use the CSV-enrichment path instead."
        ) from exc

    model_path = pose_model or (HERE.parent / "models" / "pose_landmarker_full.task")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing pose model bundle: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
    )
    rows: list[list[float]] = []
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_landmarks:
                continue
            row: list[float] = []
            for lm in res.pose_landmarks[0]:              # first pose, 33 landmarks
                vis = float(lm.visibility) if lm.visibility is not None else 0.0
                row += [lm.x, lm.y, lm.z, vis]
            rows.append(row)
    cap.release()

    df = pd.DataFrame(rows, columns=LANDMARK_COLS)
    df.insert(0, "class", label)
    return add_angle_columns(df)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _summarize(name: str, df: pd.DataFrame) -> None:
    print(f"  {name:<9} {df.shape[0]:>5} frames  ->  {df.shape[1]} cols")
    a = df[ANGLE_COLS].agg(["mean", "min", "max"]).round(1)
    print(f"      knee  mean {a.loc['mean','angle_knee']:>6}  "
          f"[{a.loc['min','angle_knee']}, {a.loc['max','angle_knee']}]")
    print(f"      hip   mean {a.loc['mean','angle_hip']:>6}  "
          f"[{a.loc['min','angle_hip']}, {a.loc['max','angle_hip']}]")
    print(f"      spine mean {a.loc['mean','angle_spine']:>6}  "
          f"[{a.loc['min','angle_spine']}, {a.loc['max','angle_spine']}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Genius landmark extraction / enrichment")
    ap.add_argument("--video", type=Path, help="extract from this video instead of the coords CSVs")
    ap.add_argument("--label", help="class label for --video frames (e.g. 'correct', 'down_deep')")
    ap.add_argument("--out", type=Path, help="output CSV path (video mode)")
    args = ap.parse_args()

    if args.video:
        if not args.label or not args.out:
            ap.error("--video requires --label and --out")
        df = extract_landmarks_from_video(args.video, args.label)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} frames -> {args.out}")
        _summarize(args.label, df)
        return

    # Default: enrich both coords CSVs.
    print(f"Enriching coords CSVs -> {PROCESSED}")
    for exercise, src in SOURCES.items():
        if not src.exists():
            print(f"  ! missing {src} — skipped")
            continue
        out = PROCESSED / f"{exercise}_enriched.csv"
        df = enrich_coords_csv(src, out)
        _summarize(exercise, df)
        print(f"      -> {out}")


if __name__ == "__main__":
    main()
