import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from z3 import *

n_vars = 2 # Number of variables
steps = 512
discretized = list(i / steps for i in range(steps // 2, steps))
xs = list(itertools.product(discretized, repeat=n_vars))

z3.set_option(precision=5)

x0 = Real('x0')
x1 = Real('x1')
x2 = Real('x2')
x3 = Real('x3')
x4 = Real('x4')
s = Solver()
s.add(0 <= x0)
s.add(x0 <= 1)
s.add(0 <= x1)
s.add(x1 <= 1)
s.add(0.5 <= x0)
s.add(0 <= x1)
s.add(0 <= x2)
s.add(((0.5 * -0.5) * -1) <= (x2 * x0))
s.add(0 <= x3)
s.add(x3 <= 1)
s.add(x0 <= x0)
s.add(x1 <= x1)
s.add(0 <= (x2 * x3))
s.add(0 <= ((x2 * x3) * x1))
s.add(0 <= (((x2 * x3) * x1) * x1))
s.add(0 <= ((x2 * x3) * x0))
s.add(0 <= (((x2 * x3) * x0) * x1))
s.add(((x2 * (-1 * x0)) + (((x2 * (1 + x0)) * (x0 * -1)) * -1)) <= ((((x2 * x3) * x0) * x1) * x1))
s.add(0 <= x4)
s.add(x4 <= 1)
s.add(x1 <= x4)
s.add(0 <= x4)
s.add(Or(x4 == x1, x4 == 0))
s.add(x0 < 1)
s.add(x1 < 1)
s.add(x3 < 1)
s.add(x4 < 1)

solutions = []

for x in tqdm(xs):
  s.push()
  s.add(x0 == x[0])
  s.add(x1 == x[1])
  if s.check() == z3.sat:
    solutions.append((x[0], x[1]))
  s.pop()

print(f"{len(solutions)} solutions found")

plt.figure(figsize=(10,10))
plt.scatter(
  list(x[0] for x in solutions),
  list(x[1] for x in solutions)
)
plt.axis([0, 1, 0, 1])
plt.show()


