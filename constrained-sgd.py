import torch
from tqdm import tqdm

n_vars = 5 # Number of variables
x = torch.full((n_vars,), 0.9, requires_grad=True)

a1 = x[0]
a2 = x[1]
c = x[2]
h0 = x[3]
h1 = x[4]

def le(lhs, rhs):
  diff = lhs - rhs
  return torch.nn.functional.relu(diff) ** 2 * 1000 + (10000 if diff > 0 else 0)

def lt(lhs, rhs):
  diff = lhs - rhs
  return torch.nn.functional.relu(diff) ** 2 * 1000 + 100 * torch.nn.functional.softplus(diff) + (10000 if diff >= 0 else 0)

def eq(lhs, rhs):
  if lhs != rhs:
    return (lhs - rhs) ** 2 + 10000
  else:
    return (lhs - rhs) ** 2
    
def disj():
  return 0

optimizer = torch.optim.Adam([x])
for iter in tqdm(range(50000)):
  ineqs = [
  le(0, x[0]),
  le(x[0], 1),
  le(0, x[0]),
  le(0, x[1]),
  le(0.5, x[2]),
  le(0, x[3]),
  le(x[3], 1),
  le(x[0], x[0]),
  le(0, (x[1] * x[3])),
  le(0, ((x[1] * x[3]) * x[0])),
  le(0, (x[2] * x[3])),
  le(((x[1] + x[2]) * 0.5), ((x[2] * x[3]) * x[0])),
  le(0, x[4]),
  le(x[4], 1),
  le(x[0], x[4]),
  le(0, x[4]),
  disj(),
  lt(x[0], 1),
  lt(x[3], 1),
  lt(x[4], 1),
  ]
  obj = 10 * (a1 + a2) # 10 * (h1 / ((1 - c) * (1 - a1) * (1 - a2)))
  loss = sum(ineqs) # + obj
  if any(ineq >= 10000 for ineq in ineqs):
    pass #print(f"{iter}: violated (loss {loss})")
  else:
    print(f"{iter}: succeeded!")
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

