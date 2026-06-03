"""Fit Genius API blueprints — one per domain (pose, session, agent).

Routes stay thin: they validate input, call the services layer, and shape the
JSON response. No TensorFlow / MediaPipe imports here.
"""
