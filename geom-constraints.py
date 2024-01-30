from ncpol2sdpa import *

n_vars = 6 # Number of variables
level = 2  # Requested level of relaxation
x = generate_variables('x', n_vars)

a1 = x[0]
a2 = x[1]
c = x[2]
h0 = x[3]
h1 = x[4]
obj = x[5]

inequalities = [
  0 <= a1,
  a1 < 1,
  0 <= a2,
  a2 < 1,
  0 <= c,
  c < 1,
  0 <= h0,
  0 <= h1,
  1 <= h1,
  1 <= 2 * a2 * c,
  h1 * a1 <= 2 * a2 * c * (h0 + h1 * a1)
]
substitutions = {(1 - c) * (1 - a1) * (1 - a2) * obj: h1}

sdp = SdpRelaxation(x)
sdp.get_relaxation(level, objective=obj, inequalities=inequalities,
                   substitutions=substitutions)

sdp.solve(solver="mosek")
print(sdp.primal, sdp.dual, sdp.status)

print(f"a1 := {sdp[a1]}")
print(f"a2 := {sdp[a2]}")
print(f"c := {sdp[c]}")
print(f"h0 := {sdp[h0]}")
print(f"h1 := {sdp[h1]}")
print(f"(1 - c) * (1 - a1) * (1 - a2) * obj := {sdp[(1 - c) * (1 - a1) * (1 - a2) * obj]}")
print(f"Objective: {sdp[obj]}")
print(f"Total mass: {sdp[h1]/sdp[(1-c)*(1-a1)*(1-a2)]}")
print(f"MSGD: D({sdp[h1]/sdp[(1-c)]}, ({sdp[a1]}, {sdp[a2]}))")

