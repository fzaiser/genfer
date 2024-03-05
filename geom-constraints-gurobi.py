import gurobipy as gp
import os

os.environ["GUROBI_HOME"] = "/home/fabian/bin/gurobi11.0.0_linux64/gurobi1100/linux64/"
os.environ["PATH"] += f":{os.environ['GUROBI_HOME']}/bin"
if "LD_LIBRARY_PATH" not in os.environ:
  os.environ["LD_LIBRARY_PATH"] = f":{os.environ['GUROBI_HOME']}/lib"
else:
  os.environ["LD_LIBRARY_PATH"] += f":{os.environ['GUROBI_HOME']}/lib"
os.environ["GRB_LICENSE_FILE"] = "/home/fabian/gurobi.lic"

model = gp.Model("PolyIneqConstraints")
model.setParam('NonConvex', 2)

a1 = model.addVar(lb=0.0, ub=1.0, name="a1")
a2 = model.addVar(lb=0.0, ub=1.0, name="a2")
c = model.addVar(lb=0.0, ub=1.0, name="c")
h0 = model.addVar(name="h0", ub=100.0)
h1 = model.addVar(name="h1", ub=100.0)
obj = model.addVar(name="obj")
a2c = model.addVar(name="a2*c")
model.addConstr(a2c == a2 * c)
h0ph1ma1 = model.addVar(name="h0+h1a1")
model.addConstr(h0ph1ma1 == h0 + h1 * a1)
_1mcm1ma1 = model.addVar(name="(1-c)*(1.0-a1)")
model.addConstr(_1mcm1ma1 == (1.0 - c) * (1.0 - a1))
_1ma2mobj = model.addVar(name="(1.0-a2)*obj")
model.addConstr(_1ma2mobj == (1.0 - a2) * obj)

model.addConstr(1.0 <= 2 * a2c)
model.addConstr(h1 * a1 <= 2 * a2c * h0ph1ma1)
model.addConstr(h1 == _1mcm1ma1 * _1ma2mobj)

model.setObjective(1.0 * obj, gp.GRB.MINIMIZE)

print(f"Is QCP? {model.isQcp}")
print(f"Is MIP? {model.isMIP}")

model.optimize()

print(f"a1 := {a1.x}")
print(f"a2 := {a2.x}")
print(f"c := {c.x}")
print(f"h0 := {h0.x}")
print(f"h1 := {h1.x}")
print(f"(1 - c) * (1 - a1) * (1 - a2) * obj := {(1 - c.x) * (1 - a1.x) * (1 - a2.x) * obj.x}")
print(f"Objective: {obj.x}")
print(f"Total mass: {h1.x/(1-c.x)*(1-a1.x)*(1-a2.x)}")
print(f"MSGD: D({h1.x/(1-c.x)}, ({a1.x}, {a2.x}))")

