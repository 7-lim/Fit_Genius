"""
train_squat.py  —  train + save the squat form/phase LSTM.

    python train_squat.py [--epochs N] [--batch-size N]

Reads backend/training/data/processed/squat_enriched.csv and writes
backend/models/squat_lstm.h5 + squat_lstm.pkl. All logic lives in lstm_common.
"""
import argparse

from lstm_common import train_exercise


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Fit Genius squat LSTM")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    train_exercise("squat", epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
