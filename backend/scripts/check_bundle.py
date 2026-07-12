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


def _find_bundle() -> str | None:
    """Find the bundle file regardless of Render's build cwd.

    Strategy (first hit wins):
      1. The list of `candidate_paths()` from `model_handler` (preferred —
         that is what the running service uses).
      2. Absolute path computed from __file__: ``<HERE>/../models/<name>``.
      3. Recursive glob from every plausible repo root for the pkl file.

    Returns the first path that exists, or None.
    """
    name = "nanocnc_model_bundle.pkl"

    # 1. Trust whatever paths model_handler already enumerates.
    for cand in candidate_paths():
        try:
            if os.path.isfile(cand):
                return cand
        except Exception:
            continue

    # 2. Walk up from the script until we find a folder that contains
    #    ``models/<name>``. This works for Render builds regardless of cwd.
    cur = os.path.abspath(HERE)
    for _ in range(6):  # up to 6 levels: scripts -> backend -> repo root
        guess = os.path.join(cur, "models", name)
        if os.path.isfile(guess):
            return guess
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # 3. Recursive glob from cwd and from BACKEND. Expensive but definitive.
    import glob
    roots = [os.getcwd(), BACKEND, "/opt/render/project/src"]
    seen = set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for hit in glob.glob(os.path.join(root, "**", name), recursive=True):
            if os.path.isfile(hit):
                return hit
    return None


def main() -> int:
    print(
        f"check_bundle: python={sys.version.split()[0]}  "
        f"numpy={np.__version__}  sklearn={sklearn.__version__}  "
        f"joblib={joblib.__version__}",
        flush=True,
    )
    print(
        f"check_bundle: cwd={os.getcwd()}  __file__={__file__}  "
        f"HERE={HERE}  BACKEND={BACKEND}",
        flush=True,
    )

    found = _find_bundle()
    if found is None:
        print(
            "!! No bundle file found. Tried candidate_paths() and recursive "
            f"glob from cwd={os.getcwd()}, BACKEND={BACKEND}. "
            f"candidate_paths()={candidate_paths()}",
            flush=True,
        )
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