import inspect
import joblib
import joblib.numpy_pickle as j

print("joblib", joblib.__version__)
print("NumpyUnpickler.__init__ sig:", inspect.signature(j.NumpyUnpickler.__init__))
print("NumpyUnpickler.__init__ source:")
print(inspect.getsource(j.NumpyUnpickler.__init__))
print()
print("NumpyUnpickler.find_class source:")
print(inspect.getsource(j.NumpyUnpickler.find_class))