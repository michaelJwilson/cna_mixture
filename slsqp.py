import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

fun = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2

# NB equality constraint means that the constraint function result is to be zero,
#    inequality means that it is to be non-negative
cons = (
    {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
    {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
    {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
)

# NB constained to be positive.
bnds = ((0.0, None), (0.0, None))

res = minimize(fun, (2, 0), method="SLSQP", bounds=bnds, constraints=cons)
exp = np.array([1.4, 1.7])
