import numpy as np
from scipy.optimize import *

nvars = 4
bounds = Bounds(
  [0, 0, 0, 0],
  [1, 1, np.inf, 1])

def constraint_fun(x):
  return [
    x[2] * x[0] - 1,
    x[2] * x[3] * x[0] * x[1] - 0.6 * x[2] * x[0] * x[0] - 0.4 * x[2]
  ]
constraints = NonlinearConstraint(constraint_fun, 0, np.inf, jac='2-point', hess=BFGS())

def objective(x):
  return x[2] * 0.6 + x[2] * 0.6 * x[0] / (1 - x[3]) / (1 - x[1])
x0 = np.array([0.98126875, 0.99990625, 1.09375, 0.99653875])

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

