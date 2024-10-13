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
- [x] Recognize cyclic `<=` constraints (nested loops)
- [ ] Turn upper bounds into bounds on the remainder besides the lower bound
- [ ] Find a way to deal with rounding errors
- [ ] scale constraints by setting the sum of coefficients to 1 (after substituting 1 for all nonlinear variables)

Bugs
====
- [x] Fix wrong second moment for die paradox example

Benchmarks
==========
- [x] investigate and fix the self-stabilization benchmarks
- [x] smaller versions of stabilization algorithms
- [x] investigate whether more benchmarks can be expressed in Polar
- [x] more nested loops
- [x] "own" -> "ours"
- [ ] more conditioning

Cleanups
========

- [ ] make handling of lower bounds consistent between residual and geometric bound semantics
- [ ] retain variable names
- [ ] Get rid of unnecessary number traits (e.g. interval only takes rationals)
