import inspect, joblib
print("joblib version:", joblib.__version__)
print("joblib.load sig:", inspect.signature(joblib.load))
# Find the inner _unpickle or load_build
print()
print("numpy_pickle module attrs:")
import joblib.numpy_pickle as np
for name in dir(np):
    if "load" in name.lower() or "pickle" in name.lower():
        print(" ", name)