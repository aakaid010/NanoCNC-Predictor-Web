"""Model handler for NanoCNC Predictor.

Loads a joblib bundle from backend/models/nanocnc_model_bundle.pkl when present.
Falls back to a simple linear-approximation demo mode so the UI still works
without a trained model.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
def _register_mt19937_aliases() -> None:
    """Make joblib.load() resolve MT19937 on every numpy layout we ship to.

    Why this exists
    ---------------
    scikit-learn Pipelines (RandomForest, GradientBoosting, ExtraTrees, ...)
    pickle the BitGenerator instance they were trained with. The MT19937
    Mersenne Twister has moved around inside numpy:

      * numpy <= 1.24 ............. ``numpy.random._mt19937.MT19937``
      * numpy 1.25 / 1.26 ......... ``numpy.random.mt19937.MT19937``
      * numpy >= 2.0  ............. ``numpy.random._mt19937.MT19937``
                                    (the underscore-prefixed module
                                    reappeared as a shim)

    When the bundle and the runtime disagree, joblib raises::

        ValueError: numpy.random._mt19937.MT19937 is not a known
                    BitGenerator module

    and ``ModelHandler`` falls back to demo mode silently. This function
    makes the unpickler happy regardless of which numpy is installed.

    The strategy is to locate MT19937 under whatever name is currently
    exposed, then register it under every legacy and modern module path
    *and* in numpy's internal ``BitGenerators`` registry, so any of the
    three layouts above can resolve the reference.
    """
    import importlib
    import types

    mt_cls = None
    for dotted in (
        "numpy.random.mt19937.MT19937",       # 1.25 / 1.26
        "numpy.random._mt19937.MT19937",      # <=1.24 and >=2.0
    ):
        try:
            module_name, _, attr = dotted.rpartition(".")
            mod = importlib.import_module(module_name)
            mt_cls = getattr(mod, attr, None)
            if mt_cls is not None:
                break
        except Exception:
            continue
    if mt_cls is None:
        # Nothing we can do; let joblib raise its own error.
        return

    # Build the alias modules so `import numpy.random._mt19937` works
    # even on numpy 1.25 / 1.26 where the file does not exist.
    try:
        _legacy = importlib.import_module("numpy.random._mt19937")
    except Exception:
        _legacy = types.ModuleType("numpy.random._mt19937")
        _legacy.MT19937 = mt_cls  # type: ignore[attr-defined]
        sys.modules.setdefault("numpy.random._mt19937", _legacy)
    else:
        if not hasattr(_legacy, "MT19937"):
            _legacy.MT19937 = mt_cls  # type: ignore[attr-defined]

    # New-path module on legacy numpy where it doesn't exist yet.
    try:
        _new = importlib.import_module("numpy.random.mt19937")
    except Exception:
        _new = types.ModuleType("numpy.random.mt19937")
        _new.MT19937 = mt_cls  # type: ignore[attr-defined]
        sys.modules.setdefault("numpy.random.mt19937", _new)
    else:
        if not hasattr(_new, "MT19937"):
            _new.MT19937 = mt_cls  # type: ignore[attr-defined]

    # Register in numpy's private BitGenerators registry that the
    # unpickler consults. Module name matters: it must match the
    # reference stored in the pickle.
    try:
        import numpy.random._pickle as _np_pickle
        for key_mod in (
            "numpy.random._mt19937",
            "numpy.random.mt19937",
        ):
            _np_pickle.BitGenerators[key_mod] = mt_cls
        # Some numpy versions also keep a flat "MT19937" entry.
        _np_pickle.BitGenerators["MT19937"] = mt_cls
    except Exception:
        pass

    # Bundles pickled under numpy 2.x reference the MT19937 class object
    # directly (e.g. ``<class 'numpy.random._mt19937.MT19937'>``). numpy
    # 1.26.4's ``__bit_generator_ctor`` only accepts a *string* module
    # name like ``'MT19937'``. Patch it to also accept the class.
    try:
        import numpy.random._pickle as _np_pickle_mod

        _orig_bgc = _np_pickle_mod.__bit_generator_ctor

        def _patched_bgc(bit_generator_name):
            if not isinstance(bit_generator_name, str):
                # numpy 2.x pickled the class object directly.
                name = getattr(bit_generator_name, "__name__", None) or type(bit_generator_name).__name__
                return _orig_bgc(name)
            return _orig_bgc(bit_generator_name)

        _np_pickle_mod.__bit_generator_ctor = _patched_bgc
    except Exception:
        pass


try:
    _register_mt19937_aliases()
except Exception:
    # Never let the shim itself break module import.
    pass


def _register_numpy_core_aliases() -> None:
    """Alias ``numpy._core.*`` to ``numpy.core.*`` for cross-version pickles.

    The shipped bundle was pickled on numpy 2.x, which stores references to
    private modules under ``numpy._core`` (e.g. ``numpy._core.numeric``).
    Render's runtime is numpy 1.26.4, which keeps those modules under
    ``numpy.core`` and does not have ``numpy._core`` at all. Alias the
    namespace so any ``import numpy._core.X`` resolves to the 1.x module
    and joblib.load can find the references.
    """
    try:
        import importlib
        import pkgutil
        import numpy.core as _core_1x
    except Exception:
        return

    # 1) Alias the parent so attribute-style lookups cascade.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = _core_1x

    # 2) Explicitly alias every submodule. Some pickles reference the
    #    full dotted path, which Python won't auto-resolve from the
    #    parent alias alone.
    try:
        _path = getattr(_core_1x, "__path__", None)
        if _path:
            for mod_info in pkgutil.iter_modules(_path):
                name = mod_info.name
                full_old = "numpy.core." + name
                full_new = "numpy._core." + name
                try:
                    sub = importlib.import_module(full_old)
                except Exception:
                    continue
                if full_new not in sys.modules:
                    sys.modules[full_new] = sub
    except Exception:
        pass


try:
    _register_numpy_core_aliases()
except Exception:
    # Never let the shim itself break module import.
    pass


def _coerce_mt19937_state(value):
    """Best-effort: coerce any incoming state into 1.26 legacy shape.

    1.26's ``MT19937.state`` validator requires the dict to be either
    ``{'bit_generator': 'MT19937', 'state': {'key': ..., 'pos': ...}}``
    or a 3-/5-tuple. numpy 2.x pickles a richer dict with one of
    these shapes:
      (A) has both ``bit_generator_state`` and legacy ``state``
      (B) ``bit_generator`` is a *class* object (not the string) and
          the legacy ``state`` lives under ``bit_generator_state``
      (C) only ``bit_generator_state`` is present
    We unwrap all of them to legacy shape.
    """
    if isinstance(value, tuple) and len(value) in (3, 5):
        return value
    if not isinstance(value, dict):
        return value
    # Already legacy?
    if (
        value.get("bit_generator") == "MT19937"
        and "state" in value
        and isinstance(value["state"], dict)
        and "key" in value["state"]
        and "pos" in value["state"]
    ):
        return value
    # 2.x shape A: outer dict contains both 2.x wrapper and legacy state
    if "bit_generator_state" in value and isinstance(value["state"], dict):
        return {
            "bit_generator": "MT19937",
            "state": value["state"],
        }
    # 2.x shape B: outer dict has class-object 'bit_generator'
    if "bit_generator" in value and isinstance(value["bit_generator"], type):
        inner = value.get("bit_generator_state") or value.get("state")
        if isinstance(inner, dict):
            return {
                "bit_generator": "MT19937",
                "state": inner,
            }
    # 2.x shape C: only bit_generator_state present
    if "bit_generator_state" in value and isinstance(value["bit_generator_state"], dict):
        return {
            "bit_generator": "MT19937",
            "state": value["bit_generator_state"],
        }
    return value


def _register_bit_generator_state_shim() -> None:
    """Patch ``joblib.numpy_pickle.NumpyUnpickler.load_build`` so that
    every ``__setstate__`` payload destined for a ``BitGenerator`` is
    massaged into the legacy 1.26 shape BEFORE the Cython validator
    sees it.

    Why this hook works
    -------------------
    * ``BitGenerator.__setstate__`` in 1.26 is a plain Python method
      ``def __setstate__(self, state): self.state = state`` — nothing
      to override there usefully.
    * ``MT19937.state`` is a Cython cdef property descriptor; Python
      ``property`` assignment cannot shadow it.
    * But pickle's BUILD opcode eventually calls
      ``obj.__setstate__(state)`` via the *unpickler*. By overriding
      ``NumpyUnpickler.load_build`` (the Python-level BUILD handler),
      we pop the state dict, coerce it, push it back, and delegate to
      the original. The Cython ``MT19937.state.__set__`` then sees
      the legacy shape and accepts it.
    """
    try:
        from joblib.numpy_pickle import NumpyUnpickler
        from numpy.random.bit_generator import BitGenerator
    except Exception:
        return

    if getattr(NumpyUnpickler, "_puku_coerce_load_build", False):
        return

    _orig_load_build = NumpyUnpickler.load_build

    def _patched_load_build(self):
        try:
            state = self.stack.pop()
        except IndexError:
            return _orig_load_build(self)
        try:
            obj = self.stack[-1]
            if isinstance(obj, BitGenerator):
                try:
                    state = _coerce_mt19937_state(state)
                except Exception:
                    # If coercion fails for any reason, leave the
                    # state as-is and let the original handler run.
                    pass
        except Exception:
            pass
        self.stack.append(state)
        return _orig_load_build(self)

    try:
        NumpyUnpickler.load_build = _patched_load_build
        NumpyUnpickler._puku_coerce_load_build = True
    except Exception:
        return


try:
    _register_bit_generator_state_shim()
except Exception:
    # Never let the shim itself break module import.
    pass


import pandas as pd


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BUNDLE_FILENAME = "nanocnc_model_bundle.pkl"
BUNDLE_PATH = os.path.join(MODELS_DIR, BUNDLE_FILENAME)
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


def candidate_paths() -> List[str]:
    """Return the list of bundle paths we will try, in priority order.

    Render runs gunicorn with rootDirectory=backend and a cwd that depends on
    how the repo was checked out. We try every plausible location so we can
    surface a useful error if nothing is found.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        BUNDLE_PATH,
        os.path.join(here, BUNDLE_FILENAME),
        os.path.join(here, "models", BUNDLE_FILENAME),
        os.path.join(here, "..", "models", BUNDLE_FILENAME),
        os.path.join(here, "..", "backend", "models", BUNDLE_FILENAME),
        os.path.join(os.getcwd(), BUNDLE_FILENAME),
        os.path.join(os.getcwd(), "models", BUNDLE_FILENAME),
        os.path.join(os.getcwd(), "backend", "models", BUNDLE_FILENAME),
    ]


class ModelHandler:
    """Loads a model bundle (or runs in demo mode) and produces predictions."""

    def __init__(self) -> None:
        self.bundle: Optional[Dict[str, Any]] = None
        self.demo_mode: bool = True
        self.last_error: Optional[str] = None
        self.bundle_path: Optional[str] = None
        self._load()

    def _load(self) -> None:
        self.last_error = None
        self.bundle_path = None
        tried: List[str] = []

        for path in candidate_paths():
            norm = os.path.normpath(path)
            tried.append(norm)
            if not os.path.isfile(norm):
                continue
            try:
                loaded = joblib.load(norm)
            except Exception as exc:  # noqa: BLE001 - we want to surface everything
                msg = "joblib.load failed at " + norm + ": " + repr(exc)
                self.last_error = msg
                print("[model_handler] " + msg, flush=True)
                continue

            if not isinstance(loaded, dict) or "length_model" not in loaded:
                keys = list(loaded.keys()) if isinstance(loaded, dict) else type(loaded).__name__
                msg = "Bundle has unexpected structure at " + norm + "; keys=" + str(keys)
                self.last_error = msg
                print("[model_handler] " + msg, flush=True)
                continue

            self.bundle = loaded
            self.demo_mode = False
            self.bundle_path = norm
            print("[model_handler] Loaded bundle from " + norm, flush=True)
            return

        # Nothing worked.
        self.demo_mode = True
        self.bundle = None
        if self.last_error is None:
            self.last_error = (
                "No bundle file found. Looked at: "
                + " | ".join(tried)
                + " | cwd=" + os.getcwd()
            )
        print("[model_handler] Demo mode active. " + self.last_error, flush=True)

    def reload(self) -> None:
        """Re-read the bundle from disk (called after /upload-model)."""
        self._load()

    # --------------------------------------------------- metadata
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

    # ----------------------------------------------- predictions
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

    # ----------------------------------------------- demo
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

