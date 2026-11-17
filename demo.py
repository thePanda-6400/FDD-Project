from pymatreader import read_mat

data = read_mat("Propeller_fault_data.mat")
for k, v in data.items():
    print(f"{k:25s} type={type(v).__name__:12s} shape={getattr(v, 'shape', None)}")

from scipy.io import loadmat
m = loadmat("Propeller_fault_data.mat", squeeze_me=True, struct_as_record=False)

import numpy as np

for name in ["CCV", "CV"]:
    d = data[name]
    print(f"\n=== {name} ===  ({len(d)} fields)")
    for k, v in d.items():
        arr = np.asarray(v) if not isinstance(v, dict) else v
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", type(v).__name__)
        print(f"  {k:20s} {str(shape):15s} {dtype}")