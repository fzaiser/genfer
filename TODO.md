# Implementation TODO

Features
===========

- [x] Ignore 0 <= ... constraints
- [x] Lower bounds
- [x] IPOPT optimizer
- [x] Adjust tolerance for linear solver
- [x] Add --objective balance option
- [x] Output marginalized bound and asymptotics
- [x] Solve problem for `-u 0` first and then re-use the decay rates and contraction factor for higher unrolling
- [ ] Turn upper bounds into bounds on the remainder besides the lower bound
- [ ] Recognize cyclic `<=` constraints (nested loops)
- [ ] Find a way to deal with rounding errors
- [ ] scale constraints by setting the sum of coefficients to 1 (after substituting 1 for all nonlinear variables)
- [ ] Add faster LP solver (HIGHS?)

Bugs
====
- [x] Fix wrong second moment for die paradox example

Benchmarks
==========
- [ ] more nested loops
- [ ] more conditioning
- [ ] smaller versions of stabilization algorithms
- [ ] investigate and fix the self-stabilization benchmarks
- [ ] investigate whether more benchmarks can be expressed in Polar

Cleanups
========

- [ ] make handling of lower bounds consistent between residual and geometric bound semantics
- [ ] retain variable names