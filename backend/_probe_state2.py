"""Trace each pickle opcode until we hit the MT19937 call."""
import pickle, sys, types, pickletools, io


class _Any:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, n):
        return _Any()

    def __setstate__(self, s):
        # Accept any state silently
        return


def _stub(name):
    return type(name, (_Any,), {"__module__": "stub"})


class TraceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            cls = super().find_class(module, name)
        except Exception:
            cls = _stub(name)
        if name == "MT19937":
            print(f">>> MT19937 found at {module}.{name}")
            orig = cls.__setstate__
            captured = {"count": 0}

            def probe(state):
                captured["count"] += 1
                n = captured["count"]
                if n <= 2:
                    print(f">>> MT19937.__setstate__ call #{n} state_type={type(state).__name__}")
                    if isinstance(state, dict):
                        print(f"    outer keys = {list(state.keys())}")
                        for k, v in state.items():
                            if isinstance(v, dict):
                                print(f"    outer[{k!r}] = dict keys={list(v.keys())}")
                                for k2, v2 in v.items():
                                    if isinstance(v2, type):
                                        print(f"      outer[{k!r}][{k2!r}] = class {v2!r}")
                                    else:
                                        s = getattr(v2, "shape", None)
                                        print(
                                            f"      outer[{k!r}][{k2!r}] = {type(v2).__name__}"
                                            + (f" shape={s}" if s is not None else "")
                                        )
                            elif isinstance(v, tuple):
                                print(f"    outer[{k!r}] = tuple len={len(v)}")
                            elif isinstance(v, type):
                                print(f"    outer[{k!r}] = class {v!r}")
                            else:
                                s = getattr(v, "shape", None)
                                print(
                                    f"    outer[{k!r}] = {type(v).__name__}"
                                    + (f" shape={s}" if s is not None else "")
                                )
                    elif isinstance(state, tuple):
                        print(f"    tuple len = {len(state)}")
                return orig(state)

            cls.__setstate__ = probe
            return cls
        return cls


# Dump first, then look for MT19937 in the pickle stream
with open("models/nanocnc_model_bundle.pkl", "rb") as f:
    data = f.read()

print("bundle size =", len(data))
print("looks like raw pickle:", data[:1] == b"\x80")

# Search for the MT19937 class reference in the stream
needle1 = b"numpy.random._mt19937"
needle2 = b"numpy.random.mt19937"
i1 = data.find(needle1)
i2 = data.find(needle2)
print("needle1 index:", i1)
print("needle2 index:", i2)
if i1 > 0:
    print("context around needle1 (-80..+120):")
    print(repr(data[max(0, i1 - 80): i1 + 120]))
if i2 > 0:
    print("context around needle2 (-80..+120):")
    print(repr(data[max(0, i2 - 80): i2 + 120]))
