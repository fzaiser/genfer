import torch
from tqdm import tqdm

n_vars = 5 # Number of variables
x = torch.zeros(n_vars, requires_grad=True)

a1 = x[0]
a2 = x[1]
c = x[2]
h0 = x[3]
h1 = x[4]

def le(lhs, rhs):
  diff = lhs - rhs
  return torch.nn.functional.relu(diff) * 1000 + (10000 if diff > 0 else 0)

def lt(lhs, rhs):
  diff = lhs - rhs
  return torch.nn.functional.relu(diff) * 1000 + torch.nn.functional.softplus(diff) + (10000 if diff >= 0 else 0)

optimizer = torch.optim.Adam([x])
for iter in tqdm(range(10000)):
  ineqs = [
    le(0, a1),
    lt(a1, 1),
    le(0, a2),
    lt(a2, 1),
    le(0, c),
    lt(c, 1),
    le(0, h0),
    le(0, h1),
    le(1, h1),
    le(1, 2 * a2 * c),
    le(h1 * a1, 2 * a2 * c * (h0 + h1 * a1)),
  ]
  obj = 10 * (a1 + a2) # 10 * (h1 / ((1 - c) * (1 - a1) * (1 - a2)))
  loss = sum(ineqs) + obj
  if any(ineq >= 10000 for ineq in ineqs):
    print(f"{iter}: violated (loss {loss})")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if iter % 10 == 0:
    print(f"{iter}: {loss}")


print(f"a1 := {a1}")
print(f"a2 := {a2}")
print(f"c := {c}")
print(f"h0 := {h0}")
print(f"h1 := {h1}")
print(f"Objective: {obj}")
print(f"Total mass: {h1/((1-c)*(1-a1)*(1-a2))}")
print(f"MSGD: D({h1/(1-c)}, ({a1}, {a2}))")

