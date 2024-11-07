# Implementation TODO

Features
===========

- [x] Ignore 0 <= ... constraints
- [x] Lower bounds
- [x] IPOPT optimizer
- [x] Adjust tolerance for linear solver
- [x] Add --objective balance option
- [x] Output marginalized bound and asymptotics
- [x] Solve problem for `-u 1` first and then re-use the decay rates and contraction factor for higher unrolling
- [x] Recognize cyclic `<=` constraints (nested loops)
- [ ] Try scaling constraints by setting the sum of coefficients to 1 (after substituting 1 for all nonlinear variables): this might improve numerical stability

Cleanups
========

- [ ] Make handling of lower bounds consistent between residual and geometric bound semantics
- [ ] Retain variable names
- [ ] Get rid of unnecessary number traits (e.g. interval only takes rationals)
