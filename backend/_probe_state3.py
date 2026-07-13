"""Walk the entire pickle byte-stream and emit every GLOBAL + BUILD pair
so we can see exactly which symbols joblib will resolve when loading the
MT19937 state."""

import io
import pickletools
import sys

with open("backend/models/nanocnc_model_bundle.pkl", "rb") as f:
    data = f.read()

print(f"Total bytes: {len(data)}")
print(f"Total opcodes: {data.count(b'(') + data.count(b'c')}")

# Find every numpy.random._mt19937 reference and dump the surrounding context
import re
needle = b"numpy.random._mt19937"
positions = [m.start() for m in re.finditer(re.escape(needle), data)]
print(f"\nnumpy.random._mt19937 occurrences: {len(positions)}")

# Find every numpy._core reference
needle2 = b"numpy._core"
positions2 = [m.start() for m in re.finditer(re.escape(needle2), data)]
print(f"numpy._core occurrences: {len(positions2)}")
for p in positions2[:10]:
    print(f"  offset={p}: {data[p:p+80]!r}")

# Now: disassemble the pickle and find all GLOBAL opcodes that mention
# MT19937 or numpy._core, with surrounding context.
print("\n--- disassembly of regions containing MT19937/numpy._core ---")
buf = io.BytesIO(data)
last_marker = 0
try:
    while True:
        pos = buf.tell()
        op = buf.read(1)
        if not op:
            break
        if op == b"c":  # GLOBAL
            module = b""
            while True:
                ch = buf.read(1)
                if ch == b"\n":
                    break
                module += ch
            name = b""
            while True:
                ch = buf.read(1)
                if ch == b"\n":
                    break
                name += ch
            if b"MT19937" in name or b"_mt19937" in module or b"numpy._core" in module:
                ctx = data[max(0, pos - 30): pos + 200]
                print(f"\n  GLOBAL @ {pos}: {module.decode()}.{name.decode()}")
                print(f"  context: {ctx!r}")
        else:
            # Skip other opcodes; we only care about GLOBAL.
            buf.seek(pos + 1)
            try:
                pickletools.dis(io.BytesIO(data[pos:pos+1]), annotate=0)
            except Exception:
                pass
except Exception as e:
    print(f"dis loop ended: {e!r}")