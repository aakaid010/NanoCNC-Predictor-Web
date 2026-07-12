"""Build-time smoke test for the model bundle.

Render runs this *during* `buildCommand`. It exits non-zero if the bundled
.pkl cannot be loaded by the pinned runtime (numpy / sklearn / joblib).
That way a broken bundle fails the deploy *before* the service tries to
serve traffic, instead of silently falling back to demo mode.

Usage
-----
    python backend/scripts/check_bundle.py

Exits
-----
    0 -> bundle loads cleanly and produces a prediction
    1 -> bundle missing
    2 -> bundle loaded but does not look like a CNC bundle
    3 -> bundle loaded but predict() raised
    4 -> bundle failed to load (MT19937 or other numpy mismatch)
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
# Make `model_handler` importable when running this script directly.
sys.path.insert(0, BACKEND)

import joblib
import numpy as np
import sklearn

from model_handler import ModelHandler, candidate_paths

BUNDLE = os.path.join(BACKEND, "models", "nanocnc_model_bundle.pkl")


def _which_exists() -> str | None:
    for cand in candidate_paths():
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    print(
        f"check_bundle: python={sys.version.split()[0]}  "
        f"numpy={np.__version__}  sklearn={sklearn.__version__}  "
        f"joblib={joblib.__version__}",
        flush=True,
    )

    found = _which_exists()
    if found is None:
        print(f"!! No bundle file found. Looked at: {candidate_paths()}", flush=True)
        return 1
    print(f"Found bundle: {found}  size={os.path.getsize(found)} bytes", flush=True)

    # 1. Direct joblib.load. If this fails the deploy must fail.
    try:
        bundle = joblib.load(found)
    except Exception as exc:  # noqa: BLE001
        print(f"!! joblib.load FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 4

    if not isinstance(bundle, dict) or "length_model" not in bundle:
        print(f"!! Bundle has unexpected structure. keys={list(bundle.keys())}", flush=True)
        return 2
    print(f"Bundle keys OK: {sorted(bundle.keys())}", flush=True)

    # 2. End-to-end through ModelHandler so we exercise the same code
    #    path the running service uses.
    handler = ModelHandler()
    if handler.demo_mode:
        print(
            f"!! ModelHandler is in DEMO MODE. last_error={handler.last_error}",
            flush=True,
        )
        return 4
    print(f"ModelHandler loaded: {handler.bundle_path}", flush=True)

    # 3. Smoke predict against a known point.
    try:
        out = handler.predict(
            cellulose_group="Wood / Pulp-based",
            acid_conc_wt_percent=60.0,
            temp_c=50.0,
            time_min=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"!! predict() raised: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 3

    print(f"Smoke predict OK: {json.dumps(out)}", flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())