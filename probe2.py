"""Test custom unpickler that bypasses BitGenerators lookup."""
import io
import pickle

import joblib
import numpy as np

print("numpy", np.__version__)

# 1) Get the LOCAL MT19937 class.
mt_cls = getattr(
    __import__("numpy.random._mt19937", fromlist=["MT19937"]),
    "MT19937",
)
print("local MT19937:", mt_cls)


class _ForgivingUnpickler(pickle.Unpickler):
    """A custom Unpickler that maps the legacy MT19937 module+name to the
    class object on the current runtime, regardless of which numpy is
    installed. This is the load-time fix; once loaded we re-dump with the
    current numpy so the bundle is self-contained.
    """

    _mt_aliases = {
        # (module, name) -> lambda returning local class
        ("numpy.random._mt19937", "MT19937"): lambda c=mt_cls: c,
        ("numpy.random.mt19937", "MT19937"): lambda c=mt_cls: c,
    }

    def find_class(self, module, name):
        key = (module, name)
        if key in self._mt_aliases:
            return self._mt_aliases[key]()
        return super().find_class(module, name)


def _load_with_shim(path):
    with open(path, "rb") as f:
        # joblib uses its own loader; bypass with plain pickle for arrays
        # but joblib.load handles the outer envelope.
        return joblib.load(path, unpickler=_ForgivingUnpickler)


try:
    bundle = _load_with_shim("/work/backend/models/nanocnc_model_bundle.pkl")
    print("Load SUCCESS, keys:", sorted(bundle.keys()))

    # Save it back as a normal pickle so it loads cleanly everywhere.
    joblib.dump(bundle, "/work/backend/models/nanocnc_model_bundle.pkl", compress=3)
    print("Re-saved bundle.")

    # Round-trip.
    bundle2 = joblib.load("/work/backend/models/nanocnc_model_bundle.pkl")
    print("Round-trip OK, keys:", sorted(bundle2.keys()))

    # Smoke-test predict.
    import pandas as pd

    feature_cols = bundle2["feature_cols"]
    row = {col: 0 for col in feature_cols}
    if "cellulose_groups" in bundle2:
        row[f"Cellulose_Group_{bundle2['cellulose_groups'][0]}"] = 1
    row["Acid_conc_wt_percent"] = 60.0
    row["Temp_C"] = 50.0
    row["Time_min"] = 60.0
    df = pd.DataFrame([row], columns=feature_cols)
    length = float(bundle2["length_model"].predict(df)[0])
    crystal = float(bundle2["crystallinity_model"].predict(df)[0])
    print(f"Predict -> length={length:.2f} nm, crystallinity={crystal:.2f} %")
except Exception as e:
    import traceback
    traceback.print_exc()