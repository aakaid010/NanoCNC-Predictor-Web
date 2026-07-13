"""Try to load the bundle with a permissive MT19937.state setter."""
import numpy as np
from numpy.random._mt19937 import MT19937

# Patch the inner Cython property's setter using __setattr__
# Cython properties live on the Cython class, not on Python type.
# Easiest: monkey-patch by replacing the class.

_orig_init = MT19937.__init__

def permissive_state_setter(self, value):
    """Accept dict or legacy tuple forms."""
    if isinstance(value, dict) and "state" in value and isinstance(value["state"], dict):
        inner = value["state"]
        if "key" in inner and "pos" in inner:
            key = np.asarray(inner["key"], dtype=np.uint32)
            if key.shape != (624,):
                raise ValueError(f"key must be 624 uint32, got {key.shape}")
            for i in range(624):
                self.rng_state.key[i] = key[i]
            self.rng_state.pos = int(inner["pos"])
            return
    if isinstance(value, tuple):
        if len(value) == 3 and value[0] == "MT19937":
            key = np.asarray(value[1], dtype=np.uint32)
            pos = int(value[2])
            for i in range(624):
                self.rng_state.key[i] = key[i]
            self.rng_state.pos = pos
            return
    if isinstance(value, dict):
        # Already in dict form, try unpacking the BitGenerator state
        bitgen = value.get("bit_generator")
        if bitgen == "MT19937" and isinstance(value.get("state"), dict):
            inner = value["state"]
            key = np.asarray(inner["key"], dtype=np.uint32)
            pos = int(inner["pos"])
            for i in range(624):
                self.rng_state.key[i] = key[i]
            self.rng_state.pos = pos
            return
    # Last resort: invoke original
    raise ValueError(f"Cannot set state from {type(value).__name__}: {value!r}")


MT19937.state = property(MT19937.state.fget, permissive_state_setter)

import joblib
try:
    bundle = joblib.load("/work/backend/models/nanocnc_model_bundle.pkl")
    print("LOAD OK, keys:", sorted(bundle.keys()))
except Exception:
    import traceback
    traceback.print_exc()