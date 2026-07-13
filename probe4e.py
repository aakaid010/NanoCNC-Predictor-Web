"""Decompress the joblib bundle and search for MT19937 references."""
import zlib
import re

with open("/work/backend/models/nanocnc_model_bundle.pkl", "rb") as f:
    data = f.read()

# Try zlib decompress
try:
    raw = zlib.decompress(data)
    print(f"zlib decompress OK: {len(raw)} bytes")
except Exception as e:
    print(f"zlib decompress failed: {e}")
    # Try skipping the first 2 bytes (zlib header)
    try:
        raw = zlib.decompress(data[2:])
        print(f"zlib decompress (skip 2) OK: {len(raw)} bytes")
    except Exception as e2:
        print(f"zlib decompress (skip 2) also failed: {e2}")
        raw = data

print(f"'MT19937' occurrences: {raw.count(b'MT19937')}")
print(f"'_mt19937' occurrences: {raw.count(b'_mt19937')}")
print(f"'mt19937' (no underscore) occurrences: {raw.count(b'mt19937') - raw.count(b'_mt19937')}")
print(f"'numpy.random' occurrences: {raw.count(b'numpy.random')}")
print(f"'BitGenerators' occurrences: {raw.count(b'BitGenerators')}")
print(f"'__bit_generator_ctor' occurrences: {raw.count(b'__bit_generator_ctor')}")

for m in re.finditer(rb"MT19937|mt19937|BitGenerators|__bit_generator_ctor", raw):
    start = max(0, m.start() - 30)
    end = min(len(raw), m.end() + 30)
    print(f"  at offset {m.start()}: {m.group()} context={raw[start:end]!r}")