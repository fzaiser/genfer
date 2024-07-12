# Tool for: "Guaranteed Bounds on Posterior Distributions of Discrete Probabilistic Programs with Loops"

## Running the benchmarks

How the benchmarks are run is explained in `benchmarks/README.md`.

## Setup

To build this tool, you need a Rust installation.
We recommend [rustup](https://rustup.rs/) to install Rust.

You need to have the following libraries/tools installed:

* Z3 needs to be installed on the system. On Ubuntu, it can be installed using `sudo apt-get install z3`
* cbc: a linear programming solver. On Ubuntu, it can be installed using `sudo apt-get install coinor-cbc coinor-libcbc-dev`
* ipopt: the nonlinear optimizer IPOPT. On Ubuntu, it can be installed using `sudo apt-get install coinor-libipopt-dev`

## Building

You can build the project using Rust's package and build manager Cargo:

```
cargo build --release
```

This may take a minute or two.
The build result is then available at `target/release/residual` and `target/release/bound`.

For development, it may be useful to run `cargo build --profile release-dev`, which performs fewer optimizations, resulting in faster compilation but slightly slower binaries.
The build result is then available at `target/release-dev/`.

## Running the tool

To run exact inference on a probabilistic program `path/to/program`, use the following command where `tool` is `residual` or `bound`:

```
target/release/tool path/to/program.sgcl
```

You can see a list of command line options as follows:

```
target/release/tool --help
```

## Probabilistic programming language

The syntax of a probabilistic program is as follows:

```
<statements>

return X;
```

where `X` is the variable whose posterior distribution will be computed and `<statements>` is a sequence of statements.

Statements take one of the following forms (where `c` is a natural number):

* constant assignment: `X := c`
* incrementing: `X += c;` or `X += a * Y + b;`
* decrementing: `X -= c;` where `c` is a natural number
* sampling: `X ~ <distribution>;` where `<distribution>` is `Bernoulli(p)`, `Categorical(p0, p1, ..., pn)` or `Geometric(p)`
* observations: `observe <event>` where `<event>` is described below. `fail` means observing an event that happens with probability 0.
* branching: `if <event> { <statements> } else { <statements> }` where `<event>` is described below
* looping: `while <event> { <statements> }`

The following distributions are supported (where `m`, `n` are natural numbers, `a`, `b` are rational numbers and `p` is a rational number between 0 and 1):

* `Bernoulli(p)`
* `Categorical(p_0, p_1, ..., p_n)`: categorical distribution on `{0, ..., n}` where `i` has probability `p_i`
* `UniformDisc(a, b)`: uniform distribution on `{a, ..., b - 1}`
* `Geometric(p)`

Rational numbers can be written as decimals (e.g. `0.4`) or fractions (e.g. `355/113`).
Events take one of the following forms:

* `n ~ <distribution>`: the event of sampling `n` from the given distribution.
**The `flip(p)` construct in the paper corresponds to `1 ~ Bernoulli(p)` in this language.**
* `X in [n1, n2, n3, ...]`: the event of `X` being in the given list of natural numbers
* `X not in [n1, n2, n3, ...]`
* `X = n`, `X != n`, `X < n`, `X <= n`, `X > n`, `X >= n` for a discrete variable `X` and a natural number `n`
* `X = Y`, `X != Y`, `X < Y`, `X <= Y`, `X > Y`, `X >= Y` for discrete variables `X` and `Y` where one of the two has finite support
* `not <event>`: negation/complement
* `<event> and <event>`: conjunction/intersection
* `<event> or <event>`: disjunction/union
