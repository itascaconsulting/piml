import pyDOE
import numpy as np
import joblib
import uuid

np.random.seed(12344)

min_E, max_E = 1e8, 5e9
assert max_E > min_E
delta_E = max_E - min_E

number_of_unknowns = 4

lhc_sizes = range(1, 18)

for lhc_size in lhc_sizes:
    print("creating cube 2**{}".format(lhc_size))
    hyper_cube = pyDOE.lhs(number_of_unknowns, 2**lhc_size)
    E_hyper_cube = min_E + hyper_cube*delta_E
    np.save(f"cube_{number_of_unknowns}_{lhc_size}.npy", E_hyper_cube)
