"""Vercel serverless entrypoint for NanoCNC Predictor.

Vercel's Python runtime picks up a WSGI callable named `app` or `model`
from any file inside `api/`. We export both for safety.

Endpoints
---------
GET  /api or /api/   - health JSON
GET  /model-info     - metadata about the loaded model (or demo mode)
POST /predict        - {cellulose_group, acid_conc_wt_percent, temp_c, time_min}
POST /upload-model   - multipart .pkl upload (writes to backend/models/)

Static frontend lives in /public and is served by Vercel's CDN at /.
See `vercel.json` for routing rules.
"""

from __future__ import annotations

import os
import sys

# Allow api/index.py to import the existing `model_handler` from `backend/`.
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(HERE, "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model_handler import BUNDLE_PATH, MODELS_DIR, ModelHandler


app = Flask(__name__)
CORS(app)

model = ModelHandler()

ALLOWED_EXTENSIONS = {".pkl"}


def _is_pkl(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/api")
@app.get("/api/")
def api_root():
    return jsonify({
        "service": "NanoCNC Predictor",
        "status": "ok",
        "demo_mode": model.demo_mode,
        "bundle_path": BUNDLE_PATH,
    })


@app.get("/model-info")
def model_info():
    info = model.get_info()
    info["demo_mode"] = model.demo_mode
    return jsonify(info)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    cellulose_group = payload.get("cellulose_group")
    try:
        acid_conc = float(payload.get("acid_conc_wt_percent"))
        temp_c = float(payload.get("temp_c"))
        time_min = float(payload.get("time_min"))
    except (TypeError, ValueError):
        return jsonify({
            "error": "acid_conc_wt_percent, temp_c, and time_min must be numbers."
        }), 400

    if not cellulose_group or not isinstance(cellulose_group, str):
        return jsonify({"error": "cellulose_group is required."}), 400

    result = model.predict(
        cellulose_group=cellulose_group,
        acid_conc_wt_percent=acid_conc,
        temp_c=temp_c,
        time_min=time_min,
    )
    return jsonify(result)


@app.post("/upload-model")
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "No file part 'model' in request."}), 400

    file = request.files["model"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _is_pkl(file.filename):
        return jsonify({"error": "Only .pkl files are accepted."}), 400

    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = secure_filename("nanocnc_model_bundle.pkl")
    save_path = os.path.join(MODELS_DIR, safe_name)
    file.save(save_path)

    model.reload()

    return jsonify({
        "message": "Model uploaded and loaded successfully.",
        "path": save_path,
        "demo_mode": model.demo_mode,
    })


model = app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
