"""
app.py  —  Fit Genius Flask entry point.

App factory: loads config, enables CORS for the Angular dev server, warms the
LSTM models once at startup (CLAUDE.md: load models once, not per request), and
registers the API blueprints.

    cd backend && python app.py            # dev server on :5000
"""
from flask import Flask, jsonify

from config import Config


def create_app(config: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)

    from flask_cors import CORS
    CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

    # Database (SQLite for dev) — create the engine + tables once.
    from db.database import init_db
    init_db(config.DATABASE_URL)

    # Register blueprints (one per domain).
    from api.agent import agent_bp
    from api.pose import pose_bp
    from api.session import session_bp
    app.register_blueprint(pose_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(agent_bp)

    @app.get("/api/health")
    def health():
        return jsonify({"data": {"status": "ok"}, "error": None})

    # Always return the { data, error } envelope, even for framework errors.
    @app.errorhandler(404)
    def _not_found(_e):
        return jsonify({"data": None, "error": "Not found."}), 404

    @app.errorhandler(413)
    def _too_large(_e):
        mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
        return jsonify({"data": None, "error": f"File too large (max {mb} MB)."}), 413

    @app.errorhandler(500)
    def _server_error(_e):
        return jsonify({"data": None, "error": "Internal server error."}), 500

    # Temp dir for video uploads (on the project drive).
    from pathlib import Path
    Path(config.UPLOAD_TMP).mkdir(parents=True, exist_ok=True)

    # Load models once, at startup.
    from services.lstm_service import warm_up
    warm_up()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=app.config["HOST"], port=app.config["PORT"],
            debug=app.config["DEBUG"])
