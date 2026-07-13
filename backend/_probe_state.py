"""Probe the actual MT19937 state dict shape stored in the bundle."""
import pickle, sys, types
import numpy as np


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()

    def __setstate__(self, s):
        if isinstance(s, tuple):
            return
        if isinstance(s, dict):
            for k, v in s.items():
                if not callable(v):
                    self.__dict__[k] = v


def _stub(name):
    return type(name, (_Any,), {"__module__": "stub"})


# Pre-register known missing names
sys.modules["Pipeline"] = _stub("Pipeline")
sys.modules["_RemainderColsList"] = _stub("_RemainderColsList")
sys.modules["_BasePCA"] = _stub("_BasePCA")
sys.modules["_PCA"] = _stub("_PCA")


class ProbeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            cls = super().find_class(module, name)
        except Exception:
            cls = _stub(name)
        if name == "MT19937":
            print("PROBE: find_class", module, name, "->", cls)
            orig = cls.__setstate__
            captured = {"called": False, "states": []}

            def probe(state):
                if not captured["called"]:
                    print("PROBE: __setstate__ type=", type(state).__name__)
                    if isinstance(state, dict):
                        print("PROBE: outer keys=", list(state.keys()))
                        for k, v in state.items():
                            if isinstance(v, dict):
                                print(f"PROBE:   outer[{k!r}] dict keys=", list(v.keys()))
                                for k2, v2 in v.items():
                                    if isinstance(v2, type):
                                        print(f"PROBE:     outer[{k!r}][{k2!r}] class=", v2)
                                    else:
                                        print(
                                            f"PROBE:     outer[{k!r}][{k2!r}] type=",
                                            type(v2).__name__,
                                            "shape=",
                                            getattr(v2, "shape", ""),
                                        )
                            elif isinstance(v, tuple):
                                print(f"PROBE:   outer[{k!r}] tuple len=", len(v))
                            elif isinstance(v, type):
                                print(f"PROBE:   outer[{k!r}] class=", v)
                            else:
                                print(
                                    f"PROBE:   outer[{k!r}] type=",
                                    type(v).__name__,
                                    "shape=",
                                    getattr(v, "shape", ""),
                                )
                    captured["called"] = True
                return orig(state)

            cls.__setstate__ = probe
            return cls
        return cls


with open("models/nanocnc_model_bundle.pkl", "rb") as f:
    try:
        ProbeUnpickler(f).load()
    except Exception as e:
        print("PROBE END:", type(e).__name__, str(e)[:300])