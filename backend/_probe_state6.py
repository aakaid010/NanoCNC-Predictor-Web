"""Stream pickletools.genops and report every GLOBAL involving MT19937/numpy._core."""

import io
import pickletools

with open("backend/models/nanocnc_model_bundle.pkl", "rb") as f:
    data = f.read()

print(f"bytes: {len(data)}")

ops = list(pickletools.genops(data))
print(f"Total opcodes: {len(ops)}")

interesting = 0
for op, arg, pos in ops:
    if op.name == "GLOBAL":
        if isinstance(arg, tuple) and len(arg) == 2:
            mod, name = arg
            mod = mod.decode("ascii", errors="replace") if isinstance(mod, bytes) else str(mod)
            name = name.decode("ascii", errors="replace") if isinstance(name, bytes) else str(name)
            if ("MT19937" in mod or "MT19937" in name or
                "_mt19937" in mod or
                "numpy._core" in mod or "numpy.core" in mod or
                "bit_generator" in name or
                "bit_generators" in mod):
                # context: 20 opcodes before & after
                interesting += 1
                idx = ops.index((op, arg, pos))
                start = max(0, idx - 6)
                end = min(len(ops), idx + 6)
                print(f"\n--- GLOBAL @{pos}: {mod}.{name} ---")
                ctx = data[pos: pos + 200]
                print(f"  bytes: {ctx!r}")
                for j in range(start, end):
                    o, a, p = ops[j]
                    al = a.decode("ascii", errors="replace") if isinstance(a, (bytes, bytearray)) else str(a)
                    print(f"    {'>' if j == idx else ' '} {p:>10}  {o.name:<22} {al!r}")

print(f"\nTotal interesting: {interesting}")