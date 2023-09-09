# Implementation TODO

Features
========

- [x] support `observe from` directly
- [x] support `X +~ distribution(Y)` directly
- [x] add a regression test suite (using expect_test)
- [x] store degree in Taylor expansion, but allow dimensions with lower degree (for finite distributions)
- [ ] specialize multivariate Taylor operations to 1d, 2d (and 3d?)
- [ ] implement polynomial multiplication in terms of fast Fourier transform
- [x] special case substituting 0
- [x] special case multiplication by (a * v^0 + b * v^1)
- [x] special case substitution of b * v for v
- [ ] special case substitution and multiplication by monomials v^a
- [ ] improve support for multivariate GFs:
  - [ ] exploit independence of program variables by maintaining a factorized representation of the generating function
  - [ ] special case substitutions v^aw^b
  - [x] allow multiplication without broadcasting
- [ ] order substitutions so that substitutions with 0 are performed first etc.
- [ ] implement O(n^2.5) algorithm for polynomial substitution (Brent-Kung 2.1)
- [ ] implement O((n log(n))^(3/2)) algorithm for polynomial substitution (Brent-Kung 2.2)
- [ ] investigate how substitution could be accelerated by using smaller dimensions for some variables (or dimension 1 for unused variables).
- [ ] add annotation nodes (to display progress during computation) to GenFun
- [ ] support expressions (involving sampling) on the RHS of assignments
- [ ] Arbitrary precision arithmetic (rug crate)
- [ ] Use variable support for bounds
- [ ] Solve incrementally for each loop rather than all at once?

Cleanups
========

- [ ] retain variable names
- [ ] use Order instead of usize for orders everywhere
