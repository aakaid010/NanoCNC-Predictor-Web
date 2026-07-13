"""Find every GLOBAL opcode (c\nmodule\nname\n) in the pickle that references
MT19937 or numpy._core and dump surrounding pickle context."""

import io
import re
import sys

with open("backend/models/nanocnc_model_bundle.pkl", "rb") as f:
    data = f.read()

print(f"Total bytes: {len(data)}")

# Use a regex tolerant of binary data inside the module/name tokens.
# Module/name CAN contain binary for non-string module names, but for our
# case they're always ASCII strings.
pat = re.compile(rb"c\n([\x20-\x7e]+)\n([\x20-\x7e]+)\n")

hits = []
for m in pat.finditer(data):
    mod = m.group(1).decode("ascii", errors="replace")
    name = m.group(2).decode("ascii", errors="replace")
    if ("MT19937" in mod or "MT19937" in name or
        "_mt19937" in mod or
        "numpy._core" in mod or "numpy.core" in mod or
        "bit_generator" in name):
        hits.append((m.start(), mod, name))

print(f"Interesting GLOBALs: {len(hits)}")
for pos, mod, name in hits:
    ctx = data[pos: pos + 240]
    print(f"\n  @{pos} GLOBAL {mod}.{name}")
    print(f"  ctx: {ctx!r}")