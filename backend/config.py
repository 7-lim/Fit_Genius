"""
config.py  —  Fit Genius backend configuration.

Reads from environment / a local .env (never committed). Sensible dev defaults
so the app boots even before .env exists (GROQ_API_KEY only matters at step 8).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load .env if python-dotenv is available and the file exists (optional).
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:  # pragma: no cover
    pass


class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = FLASK_ENV == "development"

    MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
    # as_posix() -> forward slashes, the form SQLAlchemy expects for SQLite URLs.
    DATABASE_URL = os.getenv(
        "DATABASE_URL", f"sqlite:///{(BASE_DIR / 'fitgenius.db').as_posix()}",
    )

    # Groq agent (step 8) — may be None until a key is provided.
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # CORS: Angular dev server only, in development (CLAUDE.md note 6).
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:4200")

    # Video upload (analyze-video). Temp dir kept on the project drive (D:),
    # not C:, which is space-constrained on this machine.
    UPLOAD_TMP = os.getenv("UPLOAD_TMP", str(BASE_DIR / "tmp_uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
