import numpy as np
from scipy.optimize import *

nvars = 7
bounds = Bounds(
  [0, 0, -np.inf, -np.inf, 0, 0, 0, ],
  [1, 1, np.inf, np.inf, 1, 1, 1, ])

def constraint_fun(x):
  return [
    (x[0]) - (0),
    (1) - (x[0]),
    (x[1]) - (0),
    (1) - (x[1]),
    (0) - (0),
    (0) - (0),
    (x[0]) - (0),
    (x[1]) - (0),
    (x[2]) - (0),
    ((x[2] * x[0])) - (0),
    (x[3]) - (0),
    ((x[3] * x[0])) - (0.5),
    (x[4]) - (0),
    (1) - (x[4]),
    (0) - (0),
    (0) - (0),
    (x[0]) - (x[0]),
    (x[1]) - (x[1]),
    ((x[2] * x[4])) - (0),
    (((x[2] * x[4]) * x[1])) - ((x[3] * 0.5)),
    (((x[2] * x[4]) * x[0])) - (0),
    ((((x[2] * x[4]) * x[0]) * x[1])) - (((x[3] * 0.5) * x[0])),
    ((x[3] * x[4])) - (0),
    (((x[3] * x[4]) * x[1])) - (0),
    (((x[3] * x[4]) * x[0])) - ((x[2] * 0.5)),
    ((((x[3] * x[4]) * x[0]) * x[1])) - (((x[2] * 0.5) * x[1])),
    (x[5]) - (0),
    (1) - (x[5]),
    (x[5]) - (x[0]),
    (x[5]) - (0),
    (x[6]) - (0),
    (1) - (x[6]),
    (x[6]) - (x[1]),
    (x[6]) - (0),
  ]
constraints = NonlinearConstraint(constraint_fun, 0, np.inf, jac='2-point', hess=BFGS())

def objective(x):
  return (((((x[2] * 0.5) * ((1 + (x[4] * -1)) ** -1)) + 0.5) + ((x[3] * 0.5) * ((1 + (x[4] * -1)) ** -1)))) / ((1 - 0) * (1 - 0) * (1 - x[5]) * (1 - x[6]) * 1)
x0 = np.full((nvars,), 0.9)

res = minimize(
  objective,
  x0,
  method='trust-constr',
  jac='2-point',
  hess=BFGS(),
  constraints=[constraints],
  options={'verbose': 1},
  bounds=bounds
)
print(res.x)

for i, xi in enumerate(res.x):
  print(f"x[{i}] = {xi}")

print(f"Objective: {objective(res.x)}")

