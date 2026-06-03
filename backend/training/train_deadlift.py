"""
train_deadlift.py  —  train + save the deadlift form/phase LSTM.

    python train_deadlift.py [--epochs N] [--batch-size N]

Reads backend/training/data/processed/deadlift_enriched.csv and writes
backend/models/deadlift_lstm.h5 + deadlift_lstm.pkl. All logic lives in lstm_common.
"""
import argparse

from lstm_common import train_exercise


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Fit Genius deadlift LSTM")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    train_exercise("deadlift", epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
