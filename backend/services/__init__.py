"""Fit Genius services layer — all ML / business logic lives here.

Route files never import TensorFlow or MediaPipe directly; they go through
these services (mediapipe_service, lstm_service, groq_service).
"""
