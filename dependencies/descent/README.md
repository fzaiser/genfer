# Descent

A non-linear constrained optimisation (mathematical programming) modelling
library with first and second order automatic differentiation / source-code
symbolic differentiation. Currently has an interface to the non-linear solver
[Ipopt](https://projects.coin-or.org/Ipopt).

## Dependencies

[Ipopt](https://projects.coin-or.org/Ipopt) (or
[Bonmin](https://projects.coin-or.org/Bonmin)) must be separately installed
before attempting to build as `descent_ipopt` links to the libipopt.so shared
library.

It has only been tested on linux, but presumably would also work on macos, and
potentially on windows in the right environment.

For use in your own crate, add the following to your own `Cargo.toml` file:

```toml
[dependencies]
descent = "0.3"
descent_ipopt = "0.3"
descent_macro = "0.3" # for optional but recommended expr! procedural macro use
```

## Example

The following code shows how to solve the following simple problem in IPOPT:
`min 2 y s.t. y >= x * x - x, x in [-10, 10]`.

```rust
use descent::model::Model;
use descent_ipopt::IpoptModel;

let mut m = IpoptModel::new();

let x = m.add_var(-10.0, 10.0, 0.0);
let y = m.add_var(std::f64::NEG_INFINITY, std::f64::INFINITY, 0.0);
m.set_obj(2.0 * y);
m.add_con(y - x * x + x, 0.0, std::f64::INFINITY);

let (stat, sol) = m.solve();
```

A full example of this with additional details is provided under
`descent_ipopt/examples/simple.rs`, which can be built and run as follows:

```
cargo build --release --example simple
./target/release/examples/simple
```

Code optimizations are important, so be sure to turn them on or use the release
build option when testing your code for performance.

## Modelling

The operators `+`, `-`, `*`, `.powi(i32)`, `.sin()`, and `.cos()` are supported
in mathematical expressions. Expressions can be generated either via operator
overloading or with a procedural macro (requires nightly rust).

### Automatic Differentiation via Operator Overloading

Expressions can be dynamically generated with maximum flexibility using operator
overloading. This is the approach adopted in the example provided above.

### Source-Code Transformation via Procedural Macro

If nightly rust is available, then a procedural macro can be used to "statically"
generate functions for evaluating the expression and calculating its first and
second derivatives. This provides a huge performance increase over the dynamic
operator overloading and AD approach. The above example using the procedural
macro expression generation approach looks like the following:

```rust
#![feature(proc_macro_hygiene)]

use descent::model::Model;
use descent_ipopt::IpoptModel;
use descent_macro::expr;

let mut m = IpoptModel::new();

let x = m.add_var(-10.0, 10.0, 0.0);
let y = m.add_var(std::f64::NEG_INFINITY, std::f64::INFINITY, 0.0);
m.set_obj(expr!(2.0 * y; y));
m.add_con(expr!(y - x * x + x; x, y), 0.0, std::f64::INFINITY);

let (stat, sol) = m.solve();
```

A more complete example can be found in
`descent_ipopt/examples/problem_macro.rs`.

### Parameterization

The library allows parameterisation of values and easy solver warmstarting to
enable quick model adjustments and resolving.

## TODO

- Clean up the hastily written procedural macro.
- Integrate with ipopt-sys crate and possibly ipopt crate instead of using /
  maintaining own bindings.
- Bonmin bindings (enabling MINLP).

## License

Apache-2.0 or MIT
