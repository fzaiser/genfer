# GENFER: exact Bayesian inference for discrete models via generating functions

Genfer (**GEN**erating functions to in**FER** posteriors) is a tool to compute the exact posterior distribution of discrete probabilistic models, expressed as probabilistic programs.
It is the implementation of the framework from our paper

> Zaiser, Murawski, Ong: Exact Bayesian Inference on Discrete Models via Probability Generating Functions: A Probabilistic Programming Approach. Under review. https://arxiv.org/abs/2305.17058

## Setup

To build this tool, you need a Rust installation.
We recommend [rustup](https://rustup.rs/) to install Rust.

## Building

Having installed Rust, you can build the project using Rust's package and build manager Cargo:

```
cargo build --release
```

This may take a minute or two.
The build result is then available at `target/release/genfer`.

For development, it may be useful to run `cargo build --profile release-dev`, which performs fewer optimizations, resulting in faster compilation but slightly slower binaries.
The build result is then available at `target/release-dev/genfer`.

## Running the tool

To run exact inference on a probabilistic program `path/to/program.sgcl`, use the following command:

```
target/release/genfer path/to/program.sgcl
```

You can see a list of command line options as follows:

```
target/release/genfer --help
```

## Features

* computing posterior moments (expectation, variance, skewness, kurtosis) for continuous and discrete variables
* computing posterior probability masses for discrete variables (limit can be set with `--limit`)
* sampling from continuous distributions
* sampling and observing from discrete distributions
* arbitrary precision arithmetic (via `--precision <number>`).
  This option does not guarantee that the result is correct to that precision, because rounding errors can still compound.
  But it can reduce the impact of rounding errors compared to the default precision of 53 bits.
  Together with the `--bounds` option, the actual precision of the result can be verified.
* interval arithmetic to bound the floating point rounding errors (via `--bounds`)
* rational number arithmetic (via `--rational`) to avoid rounding errors when no transcendental functions are used

## Limitations

* observations involving continuous distributions are not supported
* probability densities for posterior distributions of continuous variables cannot be computed
* the running time is exponential in the number of program variables – try to keep this number low by reusing variables! We plan on adding a program transformation pass that reduces the number of variables
* nonlinear operations on variables are not supported (for variables with finite support this can be achieved by checking for each possible value and assigning the desired result)
* unbounded loops are not supported

## Example

Consider the following problem:

> Suppose your coworker gets 10 calls per day on average and each call is a scam with probability 20%.
> By the end of today, they have only received one scam call.
> How many calls did they get in total today?

We can model the number of phone calls per day with a Poisson distribution and the number of scam calls with a Binomial distribution.
Then the question is what the posterior distribution of the number of calls is.
As a probabilistic program, this can be described as follows:

```
# File name: example.sgcl
calls ~ Poisson(10);
scams ~ Binomial(calls, 0.2);
observe(scams = 1);
return calls;
```

Genfer can compute the posterior distribution automatically and will yield an output similar to this:

```
$ target/release/genfer example.sgcl
Support is a subset of: {0, ...}

Total measure:             Z = 0.27067056647322557
Expected value:            E = 9.0
Standard deviation:        σ = 2.8284271247461876
Variance:                  V = 7.999999999999986
Skewness (3rd std moment): S = 0.35355339059330737
Kurtosis (4th std moment): K = 3.1249999999997264

Unnormalized: p(0)     = 0.0
Normalized:   p(0) / Z = 0.0
Unnormalized: p(1)     = 0.00009079985952496972
Normalized:   p(1) / Z = 0.0003354626279025117
Unnormalized: p(2)     = 0.0007263988761997578
Normalized:   p(2) / Z = 0.0026837010232200935
[...]
Unnormalized: p(25)     = 6.910972968455599e-7
Normalized:   p(25) / Z = 2.5532783481055835e-6
Unnormalized: p(n)     <= 3.1727834072246485e-7 for all n >= 26
Normalized:   p(n) / Z <= 1.1721937292869623e-6 for all n >= 26
```

## Probabilistic programming language (SGCL)

The probabilistic programming language is called SGCL (**S**tatistical **G**uarded **C**ommand **L**anguage), as it is based on pGCL (**p**robabilistic **GCL**).
The syntax of a probabilistic program is as follows:

```
<statements>

return X;
```

where `X` is the variable whose posterior distribution will be computed and `<statements>` is a sequence of statements.

Statements take one of the following forms:

* assignment: `X := a * Y + b;` or `X += a * Y + b;` where `a` and `b` are natural numbers and `X` and `Y` are program variables
* sampling: `X ~ <distribution>;` where `<distribution>` is one of the supported distributions listed below
* observations: `observe <event>` where `<event>` is described below. `fail` means observing an event that happens with probability 0.
* branching: `if <event> { <statements> } else { <statements> }` where `<event>` is described below
* bounded looping: `loop n { <statements> }` repeats the statement block `n` times where `n` has to be a natural number.

The following distributions are supported (where `m`, `n` are natural numbers, `a`, `b` are rational numbers and `p` is a rational number between 0 and 1):

* `Bernoulli(p)`
* `Bernoulli(X)` where `X` is a discrete or continuous variable taking on values between 0 and 1
* `Binomial(n, p)`
* `Binomial(X, p)` where `X` is a discrete variable
* `Categorical(p_0, p_1, ..., p_n)`: categorical distribution on `{0, ..., n}` where `i` has probability `p_i`
* `Dirac(a)` where `a`
* `Exponential(rate)` where `rate` is a rational number
* `Geometric(p)`
* `Gamma(shape, rate)` where `shape` and `rate` are rational numbers
* `NegBinomial(n, p)`
* `NegBinomial(X, p)` where `X` is a discrete variable
* `Poisson(rate)` where `rate` is a rational number
* `Poisson(rate * X)` where `rate` is a rational number and `X` is a discrete or continuous variable
* `UniformCont(a, b)`: uniform distribution on `[a, b]`
* `UniformDisc(m, n)`: uniform distribution on `{m, ..., n - 1}`

Rational numbers can be written as decimals (e.g. `0.4`) or fractions (e.g. `355/113`).
Events take one of the following forms:

* `n ~ <distribution>`: the event of sampling `n` from the given distribution
* `X in [n1, n2, n3, ...]`: the event of `X` being in the given list of natural numbers
* `X not in [n1, n2, n3, ...]`
* `X = n`, `X != n`, `X < n`, `X <= n`, `X > n`, `X >= n` for a discrete variable `X` and a natural number `n`
* `X = Y`, `X != Y`, `X < Y`, `X <= Y`, `X > Y`, `X >= Y` for discrete variables `X` and `Y` where one of the two has finite support
* `not <event>`: negation/complement
* `<event> and <event>`: conjunction/intersection
* `<event> or <event>`: disjunction/union
