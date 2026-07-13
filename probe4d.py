import inspect
import joblib.numpy_pickle as j

print("=== _unpickle ===")
print(inspect.getsource(j._unpickle))
print()
print("=== ZipNumpyUnpickler.__init__ ===")
print(inspect.getsource(j.ZipNumpyUnpickler.__init__))
print()
print("=== read_array ===")
print(inspect.getsource(j.NumpyUnpickler.read_array))
print()
print("=== load_build ===")
print(inspect.getsource(j.NumpyUnpickler.load_build))