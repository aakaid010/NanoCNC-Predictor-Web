import inspect
import joblib.numpy_pickle as j

print("=== load_compatibility ===")
print(inspect.getsource(j.load_compatibility))
print()
print("=== load ===")
print(inspect.getsource(j.load))