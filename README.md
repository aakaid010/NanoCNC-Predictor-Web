#  CNCPredict — NanoCNC Predictor

A full-stack web application that predicts the **length** and **crystallinity** of cellulose nanocrystals (CNC) produced by **sulfuric acid hydrolysis**, given the cellulose source and process parameters.

![Stack](https://img.shields.io/badge/Backend-Flask-000?logo=flask&logoColor=white)
![Stack](https://img.shields.io/badge/Frontend-Vanilla%20JS-F7DF1E?logo=javascript&logoColor=black)
![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)

---

##  Features

- **Real-time prediction** of CNC length (nm) and crystallinity (%) from 4 inputs:
  - Cellulose source group
  - H₂SO₄ concentration (wt%)
  - Hydrolysis temperature (°C)
  - Hydrolysis time (min)
- **Demo Mode** — works out of the box with a linear fallback model
- **Model upload** — drop in your own trained `.pkl` bundle via the UI
- **Training-range awareness** — warns when inputs fall outside experimental data
- **Interpretation guide** — built-in reference for result ranges
- **Zero build step** — plain HTML / CSS / vanilla JS frontend

---

##  Project Structure

```
nanocnc-predictor/
├── backend/
│   ├── app.py                 # Flask REST API
│   ├── model_handler.py       # Loads .pkl bundle; falls back to demo mode
│   ├── requirements.txt
│   └── models/
│       └── nanocnc_model_bundle.pkl   # (optional) trained model
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

##  Quick Start

### 1. Install backend dependencies

```powershell
cd backend
pip install -r requirements.txt
```

> **Python 3.13 users:** if `scikit-learn==1.4.2` fails to build, use a version you already have (e.g. `1.6.1`). The API is compatible.

### 2. (Optional) Add your trained model

In your Kaggle notebook, after training:

```python
import joblib
joblib.dump(model_bundle, "nanocnc_model_bundle.pkl")
```

Download the file and place it at `backend/models/nanocnc_model_bundle.pkl`.

The expected bundle structure:

```python
{
  "length_model":        sklearn.Pipeline,   # predicts CNC length (nm)
  "crystallinity_model": sklearn.Pipeline,   # predicts crystallinity (%)
  "metadata": {
      "training_ranges": {
          "acid_conc": [8, 75],
          "temperature": [25, 100],
          "time": [1, 900]
      },
      "feature_columns": [...],
      "model_version": "1.0",
      "trained_on": "..."
  }
}
```

> Without a `.pkl`, the app uses **Demo Mode** (linear approximation) so the UI is still testable.

### Bundle + numpy version gotcha (Render)

The trained model bundle pickles scikit-learn `Pipeline`s that internally
hold `numpy.random.MT19937` bit-generator state. The location of that
class moved between numpy releases:

| numpy version | pickled reference                |
|---------------|----------------------------------|
| ≤ 1.24        | `numpy.random._mt19937.MT19937`  |
| 1.25 – 1.26   | `numpy.random.mt19937.MT19937`   |
| ≥ 2.0         | `numpy.random._mt19937.MT19937`  |

Render pins `numpy==1.26.4`, so a bundle saved on numpy ≤ 1.24 will fail
to unpickle with `ValueError: numpy.random._mt19937.MT19937 is not a
known BitGenerator module` and the app will silently fall back to
**Demo Mode**.

Two fixes are shipped:

1. `backend/scripts/reexport_bundle.py` — re-saves the bundle against a
   modern numpy so the pickle references the layout numpy 1.26 has.
   Run it locally before pushing:

   ```bash
   cd backend
   python scripts/reexport_bundle.py
   ```

2. `backend/model_handler.py` — registers MT19937 under every layout
   so even an older pickle loads on Render.

The Render build runs `python scripts/check_bundle.py` as part of
`buildCommand`, so a broken bundle **fails the deploy** with a clear log
instead of serving demo mode silently.

### 3. Run the Flask backend

```powershell
python app.py
```

You should see:

```
* Running on http://127.0.0.1:5000
```

### 4. Serve the frontend (new terminal)

```powershell
cd ../frontend
python -m http.server 8080
```

### 5. Open in browser

| Service | URL |
| --- | --- |
| App (UI) | http://localhost:8080 |
| API | http://localhost:5000 |

---

## 🔌 API Reference

### `GET /model-info`
Returns current model status.

```json
{
  "demo": false,
  "version": "1.0",
  "trained_on": "2026-05-12",
  "training_ranges": {
    "acid_conc": [8, 75],
    "temperature": [25, 100],
    "time": [1, 900]
  }
}
```

### `POST /predict`
```json
// Request
{
  "cellulose_group": "Natural Plant Fiber",
  "acid_conc": 64,
  "temperature": 45,
  "time": 60
}

// Response
{
  "length_nm": 312.4,
  "crystallinity_pct": 78.2,
  "model_version": "1.0",
  "out_of_range": false
}
```

### `POST /upload-model`
Multipart upload of a new `.pkl` bundle. Hot-swaps the loaded model.

---

##  Dataset Context

Training ranges derived from **341 experimental literature records** of sulfuric acid hydrolysis of cellulose sources:

| Parameter | Min | Max | Typical |
| --- | --- | --- | --- |
| H₂SO₄ Concentration | 8% | 75% | 55–65% |
| Temperature | 25 °C | 100 °C | 40–65 °C |
| Time | 1 min | 900 min | 30–120 min |

---

##  Tech Stack

- **Backend:** Flask, Flask-CORS, scikit-learn, joblib, NumPy
- **Frontend:** HTML5, CSS3, vanilla JavaScript (no framework, no build step)
- **Model:** any scikit-learn `Pipeline` regressor (Random Forest, Gradient Boosting, etc.)

---

##  Development

There is no build step. To iterate:

1. Edit `frontend/app.js` / `index.html` / `style.css` → just refresh the browser
2. Edit `backend/app.py` / `model_handler.py` → restart `python app.py`

---

##  License

Research / thesis use. Please cite appropriately if used in academic work.

---


