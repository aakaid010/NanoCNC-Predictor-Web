"""Flask REST API for NanoCNC Predictor.

Frontend is served at the root ("/"), and the JSON API lives under "/api/*"
so that relative asset paths inside index.html resolve correctly without a
prefix.

Endpoints
---------
GET  /                  - serves frontend/index.html
GET  /<file>            - serves frontend/<file> (style.css, app.js, ...)
GET  /api               - health check (JSON)
GET  /api/model-info    - metadata about the loaded model (or demo mode)
GET  /api/diag         - diagnostics: bundle path, last load error, sklearn version
POST /api/predict       - {cellulose_group, acid_conc_wt_percent, temp_c, time_min}
                          -> {cnc_length_nm, crystallinity_percent, ...}
POST /api/upload-model  - multipart .pkl upload; saves to backend/models/ and reloads
"""

from __future__ import annotations

import os

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    request,
    send_from_directory,
    url_for,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

import sklearn

from model_handler import (
    BUNDLE_PATH,
    MODELS_DIR,
    ModelHandler,
    candidate_paths,
)


app = Flask(__name__, static_folder=None)
CORS(app)

handler = ModelHandler()

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")


# ---------------- root / static frontend ----------------
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def static_assets(filename: str):
    """Serve frontend static assets. 404 if missing so /api/* routes win."""
    safe = os.path.normpath(filename).lstrip(os.sep).replace("..", "")
    full = os.path.join(FRONTEND_DIR, safe)
    if not os.path.isfile(full):
        abort(404)
    return send_from_directory(FRONTEND_DIR, safe)


# ---------------- legacy redirects (keep old URLs working) ----------------
@app.get("/ui")
@app.get("/ui/")
@app.get("/ui/<path:_filename>")
@app.get("/app")
@app.get("/app/")
@app.get("/app/<path:_filename>")
def legacy_frontend(_filename: str | None = None):
    return redirect(url_for("index"), code=302)


# ---------------- API (JSON) ----------------
@app.get("/api")
@app.get("/api/")
def api_root():
    return jsonify(
        {
            "service": "NanoCNC Predictor",
            "status": "ok",
            "demo_mode": handler.demo_mode,
        }
    )


@app.get("/api/model-info")
@app.get("/model-info")
def model_info():
    info = handler.get_info()
    info["demo_mode"] = handler.demo_mode
    return jsonify(info)


@app.get("/api/diag")
@app.get("/diag")
def diag():
    """Diagnostic endpoint to debug model loading.

    Returns which candidate paths were checked, which one (if any) was
    used, whether the bundle loaded successfully, the captured error if
    not, and the installed sklearn version. Safe to expose; no secrets.
    """
    paths = []
    for cand in candidate_paths():
        norm = os.path.normpath(cand)
        exists = os.path.isfile(norm)
        paths.append(
            {
                "path": norm,
                "exists": exists,
                "size_bytes": os.path.getsize(norm) if exists else None,
            }
        )
    bundle = handler.bundle
    return jsonify(
        {
            "demo_mode": handler.demo_mode,
            "last_error": handler.last_error,
            "bundle_path": handler.bundle_path,
            "bundle_path_constant": BUNDLE_PATH,
            "bundle_loaded": bundle is not None,
            "bundle_keys": list(bundle.keys()) if bundle else None,
            "model_name_length": (bundle.get("model_name_length") if bundle else None),
            "model_name_crystallinity": (bundle.get("model_name_crystallinity") if bundle else None),
            "candidate_paths": paths,
            "cwd": os.getcwd(),
            "sklearn_version": sklearn.__version__,
        }
    )


@app.post("/api/predict")
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


# ---------------- /api/upload-model ----------------
ALLOWED_EXTENSIONS = {".pkl"}


def _is_pkl(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/api/upload-model")
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

    handler.reload()

    return jsonify(
        {
            "message": "Model uploaded and loaded successfully.",
            "path": save_path,
            "demo_mode": handler.demo_mode,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


# gunicorn entrypoint
application = app