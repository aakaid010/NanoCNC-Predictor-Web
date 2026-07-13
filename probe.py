"""Probe what the unpickle failure actually looks like inside numpy 1.26.4."""
import importlib
import sys
import types

import joblib
import numpy as np

print("numpy", np.__version__)
print("Python", sys.version.split()[0])

# What does numpy 1.26.4 actually have?
try:
    m_under = importlib.import_module("numpy.random._mt19937")
    print("numpy.random._mt19937 -> OK, MT19937 =", m_under.MT19937)
except Exception as e:
    print("numpy.random._mt19937 -> FAIL:", type(e).__name__, e)

try:
    m_plain = importlib.import_module("numpy.random.mt19937")
    print("numpy.random.mt19937 -> OK, MT19937 =", m_plain.MT19937)
except Exception as e:
    print("numpy.random.mt19937 -> FAIL:", type(e).__name__, e)

import numpy.random._pickle as p
print("BitGenerators keys (default):", list(p.BitGenerators.keys()))

# Apply the shim from model_handler.py
sys.modules.setdefault("numpy.random._mt19937", m_under)
sys.modules.setdefault("numpy.random.mt19937", types.ModuleType("numpy.random.mt19937"))
sys.modules["numpy.random.mt19937"].MT19937 = m_under.MT19937
p.BitGenerators["MT19937"] = m_under.MT19937
p.BitGenerators["numpy.random._mt19937"] = m_under.MT19937
p.BitGenerators["numpy.random.mt19937"] = m_under.MT19937
print("BitGenerators keys (after shim):", list(p.BitGenerators.keys()))

# Try to load the bundle with shim active
try:
    bundle = joblib.load("/work/backend/models/nanocnc_model_bundle.pkl")
    print("Load SUCCESS, keys:", sorted(bundle.keys()))
except Exception as e:
    import traceback
    print("Load FAILED:")
    traceback.print_exc()