"""Use joblib.numpy_pickle.NumpyUnpickler to bypass BitGenerators lookup."""
import joblib
import joblib.numpy_pickle as jnp
import numpy as np

print("numpy", np.__version__)

mt_cls = getattr(
    __import__("numpy.random._mt19937", fromlist=["MT19937"]),
    "MT19937",
)


class _ForgivingNumpyUnpickler(jnp.NumpyUnpickler):
    """Forces MT19937 to resolve to the locally available class."""

    def find_class(self, module, name):
        if name == "MT19937" and module in ("numpy.random._mt19937", "numpy.random.mt19937"):
            return mt_cls
        return super().find_class(module, name)


def _joblib_load_forgiving(path):
    with open(path, "rb") as f:
        unpickler = _ForgivingNumpyUnpickler(path, f)
        try:
            return unpickler.load()
        except Exception:
            return jnp.load(path)


try:
    bundle = _joblib_load_forgiving("/work/backend/models/nanocnc_model_bundle.pkl")
    print("Load SUCCESS, keys:", sorted(bundle.keys()))

    # Save it back under current numpy.
    joblib.dump(bundle, "/work/backend/models/nanocnc_model_bundle.pkl", compress=3)
    print("Re-saved bundle under numpy 1.26.4.")

    # Round-trip WITHOUT the shim.
    bundle2 = joblib.load("/work/backend/models/nanocnc_model_bundle.pkl")
    print("Round-trip OK (no shim), keys:", sorted(bundle2.keys()))

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
except Exception:
    import traceback
    traceback.print_exc()