"""Find every GLOBAL opcode (c\nmodule\nname\n) in the pickle that references
MT19937 or numpy._core and dump surrounding pickle context."""

import io
import pickletools
import re
import sys

with open("backend/models/nanocnc_model_bundle.pkl", "rb") as f:
    data = f.read()

# scan: every GLOBAL starts with 'c' followed by module \n name \n
i = 0
matches = []
while True:
    j = data.find(b"c\n", i)
    if j < 0:
        break
    # read module until \n
    m_start = j + 2
    m_end = data.find(b"\n", m_start)
    if m_end < 0:
        break
    module = data[m_start:m_end]
    # read name until \n
    n_start = m_end + 1
    n_end = data.find(b"\n", n_start)
    if n_end < 0:
        break
    name = data[n_start:n_end]
    matches.append((j, module, name))
    i = n_end + 1

# Print only the interesting ones
print(f"Total GLOBAL opcodes: {len(matches)}")
for pos, module, name in matches:
    if (b"MT19937" in module or b"MT19937" in name or
        b"_mt19937" in module or
        b"numpy._core" in module or b"numpy.core" in module):
        ctx_before = data[max(0, pos - 60): pos]
        ctx_after = data[pos: pos + 200]
        print(f"\nGLOBAL @ {pos}: {module.decode(errors='replace')}.{name.decode(errors='replace')}")
        print(f"  before: {ctx_before!r}")
        print(f"  after:  {ctx_after!r}")