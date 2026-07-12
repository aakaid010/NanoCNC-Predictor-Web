"""Re-export nanocnc_model_bundle.pkl against a current numpy.

The bundle shipped in this repo was originally written with an older numpy
(<= 1.24). Its pickle references `numpy.random._mt19937.MT19937`, a path
that was removed in numpy 1.25+. Render installs `numpy==1.26.4`, which means
`joblib.load` fails with:

    ValueError: numpy.random._mt19937.MT19937 is not a known BitGenerator module

Re-saving the bundle against a modern numpy makes the pickle reference
`numpy.random.mt19937.MT19937`, which numpy 1.26 *does* have. That makes the
file load cleanly on Render without needing any code changes at inference
time.

Usage
-----
    python backend/scripts/reexport_bundle.py

Inputs / outputs
----------------
Input  : backend/models/nanocnc_model_bundle.pkl
Output : backend/models/nanocnc_model_bundle.pkl  (overwritten in place)

Backup
------
The original file is copied next to it as
`backend/models/nanocnc_model_bundle.pre-reexport.bak.pkl` before being
overwritten, in case you want to compare predictions.
"""
from __future__ import annotations

import os
import shutil
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
BUNDLE = os.path.join(BACKEND, "models", "nanocnc_model_bundle.pkl")
BACKUP = os.path.join(BACKEND, "models", "nanocnc_model_bundle.pre-reexport.bak.pkl")


def main() -> int:
    if not os.path.isfile(BUNDLE):
        print(f"!! Bundle not found at {BUNDLE}. Place it there first.", flush=True)
        return 1

    print(f"numpy={np.__version__}  joblib={joblib.__version__}", flush=True)

    print(f"Loading {BUNDLE} ...", flush=True)
    bundle = joblib.load(BUNDLE)
    if not isinstance(bundle, dict) or "length_model" not in bundle:
        print("!! File does not look like a CNC bundle (missing length_model). Aborting.",
              flush=True)
        return 2

    print("Bundle keys:", sorted(bundle.keys()), flush=True)

    shutil.copy2(BUNDLE, BACKUP)
    print(f"Backup written to {BACKUP}", flush=True)

    joblib.dump(bundle, BUNDLE, compress=3)
    print(f"Re-saved bundle to {BUNDLE}", flush=True)

    # Round-trip self-check.
    reloaded = joblib.load(BUNDLE)
    expected_keys = {"length_model", "crystallinity_model"}
    if not expected_keys.issubset(reloaded.keys()):
        print(f"!! Round-trip lost keys. Got {sorted(reloaded.keys())}", flush=True)
        return 3

    # Predict smoke-test so we are sure the re-exported file still works.
    feature_cols = reloaded["feature_cols"]
    groups = reloaded.get("cellulose_groups")
    row = {col: 0 for col in feature_cols}
    if groups:
        row[f"Cellulose_Group_{groups[0]}"] = 1
    row["Acid_conc_wt_percent"] = 60.0
    row["Temp_C"] = 50.0
    row["Time_min"] = 60.0
    import pandas as pd
    df = pd.DataFrame([row], columns=feature_cols)
    length = float(reloaded["length_model"].predict(df)[0])
    crystal = float(reloaded["crystallinity_model"].predict(df)[0])
    print(f"Smoke predict -> length={length:.2f} nm, crystallinity={crystal:.2f} %",
          flush=True)

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
