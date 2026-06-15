"""Model handler for NanoCNC Predictor.

Loads a joblib bundle from `backend/models/nanocnc_model_bundle.pkl` when present.
Falls back to a simple linear-approximation demo mode so the UI still works
without a trained model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BUNDLE_PATH = os.path.join(MODELS_DIR, "nanocnc_model_bundle.pkl")

DEFAULT_CELLULOSE_GROUPS: List[str] = [
    "Wood / Pulp-based",
    "Natural Plant Fiber",
    "Agricultural Waste",
    "Processed Cellulose",
    "Algae / Marine",
    "Other",
]

# Offsets used in demo mode. Keep in sync with the spec.
DEMO_GROUP_OFFSETS: Dict[str, float] = {
    "Wood / Pulp-based": 0.0,
    "Natural Plant Fiber": 10.0,
    "Agricultural Waste": 5.0,
    "Processed Cellulose": -5.0,
    "Algae / Marine": 15.0,
    "Other": 0.0,
}


class ModelHandler:
    """Loads a model bundle (or runs in demo mode) and produces predictions."""

    def __init__(self) -> None:
        self.bundle: Optional[Dict[str, Any]] = None
        self.demo_mode: bool = True
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if not os.path.exists(BUNDLE_PATH):
            self.demo_mode = True
            self.bundle = None
            return

        try:
            loaded = joblib.load(BUNDLE_PATH)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[model_handler] Failed to load bundle: {exc}")
            self.demo_mode = True
            self.bundle = None
            return

        if not isinstance(loaded, dict) or "length_model" not in loaded:
            print("[model_handler] Bundle has unexpected structure; using demo mode.")
            self.demo_mode = True
            self.bundle = None
            return

        self.bundle = loaded
        self.demo_mode = False

    def reload(self) -> None:
        """Re-read the bundle from disk (called after /upload-model)."""
        self._load()

    # --------------------------------------------------------------- metadata
    def get_info(self) -> Dict[str, Any]:
        if self.demo_mode or self.bundle is None:
            return {
                "model_name": "Demo Mode (Linear Approximation)",
                "targets": ["cnc_length_nm", "crystallinity_percent"],
                "input_features": [
                    "Cellulose_Group_<group>",
                    "Acid_conc_wt_percent",
                    "Temp_C",
                    "Time_min",
                ],
                "r2_length": 0.0,
                "r2_crystallinity": 0.0,
            }

        return {
            "model_name": self.bundle.get(
                "model_name_length", "Unknown"
            ),
            "targets": ["cnc_length_nm", "crystallinity_percent"],
            "input_features": list(self.bundle.get("feature_cols", [])),
            "r2_length": float(self.bundle.get("r2_length", 0.0)),
            "r2_crystallinity": float(self.bundle.get("r2_crystallinity", 0.0)),
        }

    # ----------------------------------------------------------- predictions
    def predict(
        self,
        cellulose_group: str,
        acid_conc_wt_percent: float,
        temp_c: float,
        time_min: float,
    ) -> Dict[str, Any]:
        if self.demo_mode or self.bundle is None:
            return self._predict_demo(
                cellulose_group,
                acid_conc_wt_percent,
                temp_c,
                time_min,
            )

        # Real model path.
        feature_cols: List[str] = self.bundle["feature_cols"]
        cellulose_groups: List[str] = self.bundle.get(
            "cellulose_groups", DEFAULT_CELLULOSE_GROUPS
        )

        row: Dict[str, Any] = {col: 0 for col in feature_cols}
        for col in feature_cols:
            if col in ("Acid_conc_wt_percent", "Temp_C", "Time_min"):
                continue
            if col.startswith("Cellulose_Group_"):
                group = col.replace("Cellulose_Group_", "", 1)
                if group == cellulose_group and cellulose_group in cellulose_groups:
                    row[col] = 1

        row["Acid_conc_wt_percent"] = acid_conc_wt_percent
        row["Temp_C"] = temp_c
        row["Time_min"] = time_min

        df = pd.DataFrame([row], columns=feature_cols)
        length_pred = float(self.bundle["length_model"].predict(df)[0])
        crystal_pred = float(self.bundle["crystallinity_model"].predict(df)[0])

        return {
            "cnc_length_nm": round(length_pred, 2),
            "crystallinity_percent": round(crystal_pred, 2),
            "model_used": self.bundle.get(
                "model_name_length", "Trained Model"
            ),
            "confidence_note": "Prediction from trained ML model.",
        }

    # --------------------------------------------------------------- demo
    def _predict_demo(
        self,
        cellulose_group: str,
        acid_conc_wt_percent: float,
        temp_c: float,
        time_min: float,
    ) -> Dict[str, Any]:
        offset = DEMO_GROUP_OFFSETS.get(cellulose_group, 0.0)
        cnc_length = (
            800.0
            - (5.0 * acid_conc_wt_percent)
            - (2.0 * temp_c)
            + (0.1 * time_min)
            + offset
        )
        crystallinity = (
            100.0
            - (0.3 * acid_conc_wt_percent)
            - (0.1 * temp_c)
            - (0.01 * time_min)
            + offset
        )

        # Clamp to physically reasonable ranges so the demo isn't silly.
        cnc_length = float(np.clip(cnc_length, 50.0, 1500.0))
        crystallinity = float(np.clip(crystallinity, 30.0, 99.0))

        return {
            "cnc_length_nm": round(cnc_length, 2),
            "crystallinity_percent": round(crystallinity, 2),
            "model_used": "Demo Mode (Linear Approximation)",
            "confidence_note": (
                "Demo prediction using a simple linear formula. "
                "Place nanocnc_model_bundle.pkl in backend/models/ for real ML predictions."
            ),
        }
