"""Flask REST API for NanoCNC Predictor.

Endpoints
---------
GET  /model-info     - returns metadata about the loaded model (or demo mode)
POST /predict        - {cellulose_group, acid_conc_wt_percent, temp_c, time_min}
                        -> {cnc_length_nm, crystallinity_percent, model_used,
                            confidence_note}
POST /upload-model   - multipart .pkl upload; saves to backend/models/ and reloads
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model_handler import (
    BUNDLE_PATH,
    MODELS_DIR,
    ModelHandler,
)


app = Flask(__name__, static_folder=None)
CORS(app)  # CORS enabled on all routes

handler = ModelHandler()

# Path to the bundled frontend (../frontend relative to this file)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")


# ---------------------------------------------------------------- root
@app.get("/")
def index():
    # Redirect visitors to the web UI.
    return redirect("/ui", code=302)


# ---------------------------------------------------------------- health (JSON)
@app.get("/api")
@app.get("/api/")
def api_root():
    return jsonify(
        {
            "service": "NanoCNC Predictor",
            "status": "ok",
            "demo_mode": handler.demo_mode,
            "ui": "/ui",
        }
    )


# ---------------------------------------------------------------- frontend
@app.get("/ui")
@app.get("/ui/")
@app.get("/ui/<path:filename>")
def frontend(filename: str | None = None):
    """Serve the static frontend (index.html, style.css, app.js)."""
    if filename is None or filename == "":
        return send_from_directory(FRONTEND_DIR, "index.html")
    return send_from_directory(FRONTEND_DIR, filename)


# -------------------------------------------------------------- /model-info
@app.get("/model-info")
def model_info():
    info = handler.get_info()
    info["demo_mode"] = handler.demo_mode
    return jsonify(info)


# ---------------------------------------------------------------- /predict
@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    cellulose_group = payload.get("cellulose_group")
    try:
        acid_conc = float(payload.get("acid_conc_wt_percent"))
        temp_c = float(payload.get("temp_c"))
        time_min = float(payload.get("time_min"))
    except (TypeError, ValueError):
        return (
            jsonify({"error": "acid_conc_wt_percent, temp_c, and time_min must be numbers."}),
            400,
        )

    if not cellulose_group or not isinstance(cellulose_group, str):
        return jsonify({"error": "cellulose_group is required."}), 400

    result = handler.predict(
        cellulose_group=cellulose_group,
        acid_conc_wt_percent=acid_conc,
        temp_c=temp_c,
        time_min=time_min,
    )
    return jsonify(result)


# ----------------------------------------------------------- /upload-model
ALLOWED_EXTENSIONS = {".pkl"}


def _is_pkl(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/upload-model")
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "No file part 'model' in request."}), 400

    file = request.files["model"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _is_pkl(file.filename):
        return (
            jsonify({"error": "Only .pkl files are accepted."}),
            400,
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = secure_filename("nanocnc_model_bundle.pkl")
    save_path = os.path.join(MODELS_DIR, safe_name)
    file.save(save_path)

    # Reload the handler so the new model takes effect.
    handler.reload()

    return jsonify(
        {
            "message": "Model uploaded and loaded successfully.",
            "path": save_path,
            "demo_mode": handler.demo_mode,
        }
    )


if __name__ == "__main__":
    # Local development: bind to localhost on the standard port.
    app.run(host="127.0.0.1", port=5000, debug=False)


# WSGI entrypoint for production servers (gunicorn / Render)
#   gunicorn app:app
application = app
