"""Re-export nanocnc_model_bundle.pkl under Render's numpy (1.26.4).

The bundle committed to this repo was saved with numpy 2.x, which pickles the
bit-generator as a *class object* passed to ``__bit_generator_ctor``. Render
runs numpy 1.26.4, whose ``__bit_generator_ctor`` requires a *string* name like
``'MT19937'``. So ``joblib.load`` on Render explodes with:

    ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known
    BitGenerator module.

This script:
  1. Monkey-patches numpy 1.26.4's ``__bit_generator_ctor`` to also accept a
     class object (the numpy 2.x format).
  2. Loads the bundle.
  3. Re-dumps it. The re-dumped pickle now uses numpy 1.26.4's native string
     format, so Render loads it with zero shims.

Run this *once* inside a Python 3.11 + numpy 1.26.4 + scikit-learn 1.5.2 +
joblib 1.4.2 + pandas 2.2.1 environment (see ``Dockerfile.reexport``), then
commit the resulting ``nanocnc_model_bundle.pkl``.

Usage:
    python backend/scripts/reexport_bundle.py [BUNDLE_PATH]
"""
from __future__ import annotations

import os
import shutil
import sys
import types

import joblib
import numpy as np
import numpy.random._pickle as _np_pickle

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
DEFAULT_BUNDLE = os.path.join(BACKEND, "models", "nanocnc_model_bundle.pkl")


def _patch_numpy_legacy_pickle() -> None:
    """Make numpy 1.26.4 understand the numpy 2.x pickle format.

    Two adjustments are needed:

    1. ``__bit_generator_ctor`` was changed to accept a *class object* in
       numpy 2.x. numpy 1.26.4 still insists on a string name like
       ``'MT19937'``. We monkey-patch it to accept either form.

    2. Bundles pickled under numpy 2.x reference ``numpy._core.*`` modules
       (e.g. ``numpy._core.numeric``). numpy 1.26.4 exposes them as
       ``numpy.core.*``. We register stub submodules so ``import`` works.
       After re-dumping under numpy 1.26.4, these stubs go away.
    """
    # --- 1. BitGenerator class-object compatibility ----------------------------
    cls_to_name = {cls: name for name, cls in _np_pickle.BitGenerators.items()}

    def __bit_generator_ctor(bit_generator_name="MT19937"):
        if bit_generator_name in _np_pickle.BitGenerators:
            return _np_pickle.BitGenerators[bit_generator_name]()
        if isinstance(bit_generator_name, type) and bit_generator_name in cls_to_name:
            return bit_generator_name()
        raise ValueError(
            f"{bit_generator_name!r} is not a known BitGenerator module."
        )

    _np_pickle.__bit_generator_ctor = __bit_generator_ctor
    import numpy.random as _nr
    _nr.__bit_generator_ctor = __bit_generator_ctor
    if hasattr(_nr, "__randomstate_ctor"):
        _nr.__randomstate_ctor = _np_pickle.__randomstate_ctor
    if hasattr(_nr, "__generator_ctor"):
        _nr.__generator_ctor = _np_pickle.__generator_ctor

    # --- 2. numpy._core.* -> numpy.core.* alias modules ------------------------
    import sys
    import types
    import numpy as _np

    for sub in ("_core",):
        full = f"numpy.{sub}"
        if full in sys.modules:
            continue
        stub = types.ModuleType(full)
        stub.__path__ = []  # mark as a package
        sys.modules[full] = stub
        setattr(_np, sub, stub)
    # Bridge common submodules that may be referenced in numpy 2.x pickles.
    _bridge_aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.fromnumeric": "numpy.core.fromnumeric",
        "numpy._core.shape_base": "numpy.core.shape_base",
        "numpy._core._methods": "numpy.core._methods",
    }
    for alias_name, target_name in _bridge_aliases.items():
        try:
            target = __import__(target_name, fromlist=["*"])
        except ImportError:
            continue
        if alias_name == "numpy._core":
            # Replace the bare stub with a proxy that exposes the real module's
            # attributes on attribute access.
            class _Proxy(types.ModuleType):
                def __init__(self, name, real):
                    super().__init__(name)
                    self.__dict__["_real"] = real

                def __getattr__(self, name):
                    return getattr(self.__dict__["_real"], name)

            sys.modules[alias_name] = _Proxy(alias_name, target)
        else:
            sys.modules[alias_name] = target


def _resolve_bundle_path() -> str:
    if len(sys.argv) > 1 and sys.argv[1]:
        return os.path.abspath(sys.argv[1])
    env = os.environ.get("BUNDLE_PATH")
    if env:
        return os.path.abspath(env)
    return DEFAULT_BUNDLE


def main() -> int:
    bundle_path = _resolve_bundle_path()
    if not os.path.isfile(bundle_path):
        print(f"!! Bundle not found at {bundle_path}.", flush=True)
        return 1

    print(f"numpy={np.__version__}  joblib={joblib.__version__}", flush=True)

    _patch_numpy_legacy_pickle()

    print(f"Loading {bundle_path} ...", flush=True)
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or "length_model" not in bundle:
        print("!! File does not look like a CNC bundle (missing length_model).", flush=True)
        return 2

    print("Bundle keys:", sorted(bundle.keys()), flush=True)

    backup_path = bundle_path + ".pre-reexport.bak.pkl"
    shutil.copy2(bundle_path, backup_path)
    print(f"Backup written to {backup_path}", flush=True)

    joblib.dump(bundle, bundle_path, compress=3)
    print(f"Re-saved bundle to {bundle_path}", flush=True)

    reloaded = joblib.load(bundle_path)
    expected = {"length_model", "crystallinity_model"}
    if not expected.issubset(reloaded.keys()):
        print(f"!! Round-trip lost keys. Got {sorted(reloaded.keys())}", flush=True)
        return 3

    # Smoke predict.
    import pandas as pd

    feature_cols = reloaded["feature_cols"]
    groups = reloaded.get("cellulose_groups")
    row = {col: 0 for col in feature_cols}
    if groups:
        row[f"Cellulose_Group_{groups[0]}"] = 1
    row["Acid_conc_wt_percent"] = 60.0
    row["Temp_C"] = 50.0
    row["Time_min"] = 60.0
    df = pd.DataFrame([row], columns=feature_cols)
    length = float(reloaded["length_model"].predict(df)[0])
    crystal = float(reloaded["crystallinity_model"].predict(df)[0])
    print(f"Smoke predict -> length={length:.2f} nm, crystallinity={crystal:.2f} %", flush=True)

    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
