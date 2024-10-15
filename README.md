# Artifact for: "Guaranteed Bounds on Posterior Distributions of Discrete Probabilistic Programs with Loops"

## Abstract

This is the artifact for "Guaranteed Bounds on Posterior Distributions of Discrete Probabilistic Programs with Loops" (POPL 2025).
The paper proposes two new methods to bound the posterior distribution of probabilistic programs.
This artifact contains the implementation of the two semantics from the paper (Residual Mass Semantics & Geometric Bound Semantics).
It also includes the benchmarks from the paper and scripts to reproduce the reported data (i.e. plots and tables).
The artifact is distributed as a VirtualBox image to avoid issues with dependencies.

To get started, follow the artifact URL and download the README file.
Then follow the instructions in the README.



## 1 Instructions for Running the Virtual Machine

### 1.1 Download

* Download the VM image `popl2025.ova` from https://doi.org/10.5281/zenodo.13935838.
* Download Oracle VirtualBox 7.1.2 for your system here: https://www.virtualbox.org/wiki/Downloads.



### 1.2 Setup

* Import the VM image `popl2025.ova` and start the virtual machine.
* The guest OS is Ubuntu 22.04.05.
* Log in with user name `user` and password `123`.
* All dependencies are already installed on the VM.
* Guest extensions are installed, so you should be able to copy and paste between the host and the guest system.
* Make sure that the correct keyboard layout ("en": English US), is selected (top right of the screen, next to the network indicator etc.).

**Hardware requirements**: The VM was tested on a laptop computer with a 13th Gen Intel® Core™ i7-1365U × 12 processor and 16.0 GB RAM, running Ubuntu 22.05.05.
Similar hardware should also be able to run this artifact.



### 1.3 Directory Layout

The artifact is located at `~/artifact`.
It contains three directories:

- `tool/`: the most important subdirectory.
  It includes the source code of our tool, as well as the benchmarking and evaluation scripts.
  - `benchmarks/`: contains the benchmark programs in our format (`.sgcl`) and some also in Polar's format (`.prob`).
  - `dependencies/`: contains two libraries that had to be patched.
    The remaining dependencies are from crates.io, the usual Rust package registry.
  - `src/`: the source code of our tool (details below).
  - `target/`: build artifacts of our tool.
- `gubpi/`: the source code of the tool GuBPI (Beutner et al., PLDI 2022) for comparison.
- `polar/`: the source code of the tool Polar (Moosbrugger et al., OOPSLA 2022) for comparison.
  It was slightly extended to be able to handle the geometric distribution, which we need.
  For the changes, see https://github.com/probing-lab/polar/pull/28.





## 2 Trying out the tool

Our tool is written in Rust and can be built with Cargo as usual.
The evaluation scripts are written in Python.

To try out our tool, you can run it on a simple example, e.g. on the Die Paradox example from the paper (`tool/benchmarks/die_paradox.sgcl`).

Open a terminal by pressing `Ctrl-Alt-T` or right-click and select `Open in Terminal` inside the `~/artifact/tool` folder.

```shell
cd artifact/tool # if you're not already in the tool directory
cargo build --release --bins # build the tool

target/release/residual benchmarks/die_paradox.sgcl -u 30
```

This runs the Residual Mass Semantics with unrolling limit 30 and should produce the following output:

```
Probability masses:
p(0) = 0
p(1) ∈ [0.6666666666666374, 0.6666666666667347]
p(2) ∈ [0.2222222222222125, 0.22222222222228374]
p(3) ∈ [0.07407407407407082, 0.07407407407413344]
p(4) ∈ [0.024691358024690278, 0.024691358024750004]
p(5) ∈ [0.00823045267489676, 0.008230452674955523]
p(6) ∈ [0.002743484224965586, 0.00274348422502403]
p(7) ∈ [0.0009144947416551954, 0.0009144947417135321]
p(8) ∈ [0.00030483158055173183, 0.00030483158061003285]
p(9) ∈ [0.00010161052685057727, 0.00010161052690886643]
p(10) ∈ [0.00003387017561685909, 0.0000338701756751443]
p(11) ∈ [0.00001129005853895303, 0.00001129005859723692]
p(12) ∈ [3.763352846317677e-6, 3.7633529046011254e-6]
p(13) ∈ [1.2544509487725588e-6, 1.2544510070558613e-6]
p(14) ∈ [4.181503162575196e-7, 4.1815037454077305e-7]
p(15) ∈ [1.3938343875250653e-7, 1.3938349703574368e-7]
p(16) ∈ [4.6461146250835515e-8, 4.6461204534067224e-8]
p(17) ∈ [1.5487048750278505e-8, 1.5487107033508404e-8]
p(18) ∈ [5.162349583426168e-9, 5.1624078666554656e-9]
p(19) ∈ [1.7207831944753894e-9, 1.7208414777044853e-9]
p(n) ∈ [0.0, 5.828322899542719e-14] for all n >= 20

Moments:
0-th (raw) moment ∈ [0.9999999999999417, 1.0000000000000584]
1-th (raw) moment ∈ [1.49999999999949, inf]
2-th (raw) moment ∈ [2.9999999999863034, inf]
3-th (raw) moment ∈ [8.249999999585205, inf]
4-th (raw) moment ∈ [29.99999998732587, inf]
Total time: 0.01112s
```

As you can see, the Residual Mass Semantics found bounds on the probability masses and lower bounds on the moments, but cannot find nontrivial upper bounds on the moments.
For the latter, we need the Geometric Bound Semantics:

```shell
target/release/geobound benchmarks/die_paradox.sgcl -u 30 --objective ev
```

This command runs the Geometric Bound Semantics with an unrolling limit of 30 and instructs the tool to optimize the bounds of the expected value (`ev`).
It produces some more detailed output with information about the constraint generation, solution, and optimization process, which ends with:

```
[...]

Probability masses:
p(0) = 0.0
p(1) ∈ [0.6666642308420133, 0.6666673132400747]
p(2) ∈ [0.22222141028067113, 0.2222227543769802]
p(3) ∈ [0.07407380342689038, 0.07407451205788756]
p(4) ∈ [0.024691267808963458, 0.024691718502199346]
p(5) ∈ [0.008230422602987819, 0.008230749361737594]
p(6) ∈ [0.0027434742009959396, 0.00274372840963568]
p(7) ∈ [0.0009144914003319799, 0.0009146957150214882]
p(8) ∈ [0.00030483046677732666, 0.0003049969893546323]
p(9) ∈ [0.00010161015559244222, 0.00010174666465124123]
p(10) ∈ [0.0000338700518641474, 0.00003398222225793997]
p(11) ∈ [0.000011290017288049134, 0.00001138227722998843]
p(12) ∈ [3.763339096016378e-6, 3.839252372571109e-6]
p(13) ∈ [1.2544463653387928e-6, 1.3169191753416838e-6]
p(14) ∈ [4.1814878844626425e-7, 4.6956406708469844e-7]
p(15) ∈ [1.393829294820881e-7, 1.8169893127245337e-7]
p(16) ∈ [4.646097649402936e-8, 8.12884225805633e-8]
p(17) ∈ [1.5486992164676452e-8, 4.4151235418448015e-8]
p(18) ∈ [5.162330721558818e-9, 2.8754076641518152e-8]
p(19) ∈ [1.7207769071862728e-9, 2.1137681082653185e-8]
[...]

Asymptotics: p(n) <= 7.855965656259541e-7 * 0.8230384253075658^n for n >= 31

Moments:
0-th (raw) moment ∈ [0.9999963462630153, 1.0000036537503347]
1-th (raw) moment ∈ [1.4999945193943771, 1.5000206471407989]
2-th (raw) moment ∈ [2.9999890387842374, 3.0002127045826996]
3-th (raw) moment ∈ [8.249969856517756, 8.253276566000135]
4-th (raw) moment ∈ [29.99989038308604, 30.067296968529845]
Total time: 0.12318 s
```

In particular, it finds the bound `[1.4999945193943771, 1.5000206471407989]` on the expected value (i.e. the `1-th (raw) moment`).

You can also optimize the asymptotic tail bound instead (for this, no unrolling is needed):

```
$ target/release/geobound benchmarks/die_paradox.sgcl --objective tail
[...]
Asymptotics: p(n) <= 674.6678732164137 * 0.33942394950033866^n for n >= 9
[...]
Total time: 0.07913 s
```

As you can see, the asymptotic tail bound of `O(0.33942394950033866^n)` is much better than the asymptotic tail bound of `O(0.8230384253075658^n)` obtained above with a different optimization objective.

All the flags that the tool accepts are documented in the help text (`target/release/geobound --help`).
More information on how to use this tool can be found further down.






## 3 Evaluation instructions

### 3.1 Claims

We make the following claims in the paper:

* **Claim 1 (Section 6.1)**: The geometric Bound Semantics is applicable often (over 80% of the benchmarks).
* **Claim 2 (Section 6.2)**: The Geometric Bound Semantics typically yields good bounds (majority of the benchmarks).
* **Claim 3 (Section 6.3)**: The Geometric Bound Semantics is usually faster and applicable to more benchmarks than Polar (another tool to analyze the moments of probabilistic loops).
* **Claim 4 (Section 6.3)**: The Residual Bound Semantics as implemented in our tool is orders of magnitude faster than GuBPI (another tool to compute guaranteed bounds for probabilistic programs).
* **Claim 5 (Section 6.4)**: There are trade-offs between the two semantics presented in the paper: while the Geometric Bound Semantics is much more informative, the Residual Mass Semantics is much faster in practice.

The following instructions describe how to verify those claims.



### 3.2 Generating data for the remaining claims:

For the first 3 claims, we have to run our tool (and the Polar tool, for comparison) on about 40 benchmarks in `tool/benchmarks/**/*.sgcl`.
The benchmarks were collected from the repositories of the tools Polar, Prodigy, and PSI, as well as some benchmarks added by us (see details in the paper).
This will take a few hours (it took 2 hours on my computer).

**If you don't want to wait a long time**, you can skip this step at first.
We have already included the resulting data of the benchmarks in `bench-results.json`, so you can directly continue with the remaining steps.

```shell
cd tool/benchmarks # if not already
# WARNING: This will take a few hours. You can skip this step at first, if you want.
python3 bench.py
# Writes a file `bench-results.json`.
```

This script runs each benchmark with the following configurations:

* `geobound -u 0`: Geometric Bound Semantics without unrolling and without optimization.
  This checks whether a geometric bound can be found at all.
  It runs the command `../target/release/geobound <benchmark>.sgcl -u 0`.
* `geobound -u 30 --objective ev`: Geometric Bound Semantics with unrolling limit 30 and optimizing the bound on the expected value of the program distribution.
  To obtain good bounds on the expected value, unrolling is needed (`-u 30`) and the optimization objective must be set (`--objective ev`).
* `geobound -u 0 --objective tail`: Geometric Bound Semantics without unrolling but optimizing the tail asymptotic bound.
  For tail bounds, unrolling is not helpful (except for numerical issues, in some cases), so the unrolling limit is set to 0.
* `polar`: The Polar tool by Moosbrugger et al. (OOPSLA2022), see https://github.com/probing-lab/polar.
  Polar cannot compute tail bounds but can compute exact moments (in particular expected values) for some benchmarks.
  We have translated the relevant benchmarks to Polar's format (file extension: `.prob`).

Each configuration is run 3 times for each benchmark, and the fastest run is recorded.
(This is to mitigate noise in the running times due to background activity on the computer.)
Some benchmarks require slight adjustments to the unrolling limit or contraction invariant size.
These changes are listed at the top of the file (e.g. `# flags(geobound): ...`) and parsed by the benchmarking script.

The results are written to `bench-results.json` and will be visualized in the following steps.



### 3.3 Claim 1: Geometric Bound Semantics is applicable often (Table 2)

To generate Table 2 (Applicability of the Geometric Bound Semantics), run this script:

```shell
cd tool/benchmarks # if not already
python3 tables.py applicability
```

This script reads `bench-results.json` and outputs a LaTeX table containing the results.
For each benchmark, it records some statistics and lists whether a bound could be found, and if so, the time it took.

One can see that for over 80% of the benchmarks, the Geometric Bound Semantics succeeds, supporting our claim about its applicability.



### 3.4 Claim 2: Geometric Bound Semantics yields useful bounds (Table 3)

To generate Table 3 (Quality of the Geometric Bounds), run this script:

```shell
cd tool/benchmarks # if not already
python3 tables.py quality
```

It reads `bench-results.json` and outputs a LaTeX table containing the results.

One can see that the upper and lower bounds on the expected value are usually close together and all the bounds are nontrivial.
Most of the tail bounds are also very close to the theoretical optimum (where the latter is known, see Table 3 in the paper).
This supports the claim that our implementation of the Geometric Bound Semantics yields useful bounds.



### 3.5 Claim 3: Geometric Bound Semantics is typically faster and more often applicable than Polar (Table 5)

To generate Table 5 (Comparison of Geometric Bounds and Polar), run this script:

```shell
cd tool/benchmarks # if not already
python3 tables.py polar-comparison
```

It reads `bench-results.json` and outputs a LaTeX table containing the results.

One can see that our tool is typically faster than Polar and applicable to more benchmarks.
The computed bounds are typically very close to the exact values.



### 3.6 Claim 4: Our Residual Mass Semantics is orders of magnitude faster than GuBPI (Table 4)

We compare our tool with GuBPI on three benchmarks: the geometric counter, asymmetric random walk, and die paradox example from the paper.
We translated the examples to GuBPI's file format (`.spcf`) and you can run GuBPI on them as follows.

```shell
cd tool/benchmarks # if not already
../../gubpi/app/GuBPI geo.spcf # ~1 second
../../gubpi/app/GuBPI asym_rw.spcf # ~90 seconds
../../gubpi/app/GuBPI die_paradox.spcf # ~180 seconds
```

The running time is reported by GuBPI at the end of each run.
The computed bounds are saved in `output/<benchmark>-norm.bounds`.
Note that GuBPI has trouble normalizing the bounds for `die_paradox.spcf`, so we only considered the usable lower bounds in `output/die_paradox-unnorm.bounds`.

To run our tool with the Residual Mass Semantics on the same examples with comparable settings, use these commands:

```shell
cd tool/benchmarks # if not already
../target/release/residual geo.sgcl -u 100 # ~0.006 seconds
../target/release/residual asym_rw.sgcl -u 14 # ~0.002 seconds
../target/release/residual die_paradox.sgcl -u 6 # ~0.0008 seconds
```

The running time is reported by our tool at the end of each run.
You can check that the bounds of our tool are at least as good as GuBPI's by comparing our tool's output in the terminal with GuBPI's output in `output/<benchmark>-norm.bounds`.
Note that for the `die_paradox` example, GuBPI fails to compute normalized bounds, so we compare our tool's unnormalized bounds (under `Unnormalized bounds:` in the output) with GuBPI's unnormalized bounds in `output/die_paradox-unnorm.bounds`.

These experiments support the data in Table 4 and demonstrate that our tool is orders of magnitude faster than GuBPI to produce the same (or better) bounds.



### 3.7 Claim 5: Tradeoffs between the two semantics (Table 6, Fig. 7)

Next, let's reproduce the plots from Fig. 7 (Section 6.4) in the paper.
To do this, run the following commands:

```shell
cd tool/benchmarks # if not already
# Generate the data for the plots (takes about 1min):
./comparison.sh
# Plot the data:
./plots.sh
# Now the plots (named `plot_<benchmark>.pdf`) are in the `benchmarks/` directory
```

The script `./comparison.sh` runs the Geometric Bound Semantics on 5 benchmarks: `asym_rw.sgcl`, `coupon-collector.sgcl`, `die_paradox.sgcl`, `geo.sgcl`, `herman.sgcl`.
Each benchmark is run 4 times: once with the Residual Mass Semantics and three times with the Geometric Bound Semantics, each time with a different optimization objective (`total` for probability masses, `ev` for moments, or `tail` for tail asymptotics).
The output of each run is written to `outputs/<benchmark>-residual.txt` for the Residual Mass Semantics and `outputs/<benchmark>-bound-<objective>.txt` for the Geometric Bound Semantics.
The data for Table 6 is taken from the relevant files in `outputs/`.

The claim here is a bit subtle:
The residual mass semantics is faster and yields tighter bounds for the probability masses on small values, but the bound is flat, i.e. the difference between upper and lower bounds on probability masses is constant.
In contrast, the geometric bound semantics yields a decreasing bound and can bound moments and tail asymptotics in addition to probability masses.
For details on this claim, see Section 6.4 in the paper.






## 4 Further details

Here is some additional information about how to use the artifact.
As a starting point for experimentation, have a look at the `*.sgcl` files in `benchmarks`, e.g. `benchmarks/geo.sgcl`.

## 4.1 How to use the tool

The tool consists of two binaries (created in `target/release` after running `cargo build --release --bins`): `residual` and `geobound`.

The `residual` binary implements the Residual Mass Semantic (Section 3 in the paper).
It computes bounds on probability masses of the posterior distribution by unrolling all loops a number of times and bounding the remaining probability mass.
It takes the following command-line arguments:

* `<filename>.sgcl`: a file containing the probabilistic program to analyze.
  The probabilistic programming language is described below.
* `-u <number>` or `--unroll <number>` (default: 8): the loop unrolling limit (i.e. number of times each loop is unrolled).
  Higher values take longer, but yield more precise results.
* `--limit <number>`: the limit up to which probability mass bounds are output, e.g. `--limit 50` outputs `p(0), ..., p(50)`.

The `geobound` binary implements the Geometric Bound Semantics (Section 4 and 5 in the paper).
It computes a global bound on the program distribution that takes the form of an EGD (eventually geometric distribution).
In order to find such an EGD bound, it needs to synthesize a contraction invariant, a problem that the Geometric Bound Semantics reduces to a system of polynomial inequality constraints.
Typically, we do not just want any solution to this constraint problem, but one that minimizes a certain bound, e.g. the expected value or the tail asymptotics.
If such an objective is specified, the tool will try to minimize this objective.
It takes the following command-line arguments:

* `<filename>.sgcl`: a file containing the probabilistic program to analyze.
  The probabilistic programming language is described below.
* `-u <number>` or `--unroll <number>` (default: 8): the loop unrolling limit (i.e. the number of times each loop is unrolled).
  Higher values take longer, but (usually) yield more precise results (unless the solver encounters numerical issues).
* `-d <number>` (default: 1): the size of the contraction invariant to be synthesized.
  The default of 1 is usually fine, but some programs only admit larger contraction invariants.
  Increasing this value can also sometimes improve the bounds.
* `--objective <objective>` (default: none): the bound to minimize.
  It can be one of the following:
  * `total`: the total probability mass
  * `ev`: the expected value
  * `tail`: the tail asymptotic bound, i.e. the `c` in `O(c^n)` where `p(n) = O(c^n)` as `n` grows.
* `--solver <solver>` (default: `ipopt`): the solver to use for the constraint problem.
  The following solvers are available:
  * `ipopt`: the IPOPT solver. This is a fast numerical solver and almost always the best option.
  * `z3`: the Z3 SMT solver. This is an exact solver but only works for small programs and low unrolling limits.
    On the upside, it can (in principle) prove infeasibility, i.e. that no bound exists for the given invariant size.
* `--optimizer <optimizer>` (default: `ipopt adam-barrier linear`): a list of optimizers to minimize the objective.
  The optimizers are run in the order they are specified, e.g. `--optimizer ipopt --optimizer linear`.
  The following optimizers are available:
  * `ipopt`: the IPOPT solver. This is a fast numerical optimizer and should usually be included.
  * `linear`: a linear programming solver (uses COIN-CBC).
    This is an extremely fast way to optimize the linear variables of the constraint problem.
    It should usually be included, but is not enough on its own, as it does not touch the nonlinear variables.
  * `adam-barrier`: a solver provided by us that combines the barrier method with the ADAM optimizer.
    This is usually the slowest solver, so it is best omitted in some cases by passing `--optimizer ipopt --optimizer linear`.
    However, it is typically useful for `--objective tail`, which is why it is included in the default.
* `--keep-while`: deactivates the usual transformation of `while` loops into `do-while` loops.
  By default, the semantics conceptually treats `while <event> { ... }` as `if <event> { do { ... } while <event> }`.
  (But note that this is not valid syntax!)
  Occasionally, it can be useful to deactivate this transformation, which is what `--keep-while` does.



## 4.2 Probabilistic programming language (SGCL)

The syntax of a probabilistic program is as follows:

```
<statements>

return X;
```

where `X` is the variable whose posterior distribution will be computed and `<statements>` is a sequence of statements.

**Statements** take one of the following forms (where `c` is a natural number):

* constant assignment: `X := c`
* incrementing: `X += c;` or `X += Y;` (the latter is only supported for `Y` with finite support)
* decrementing: `X -= c;` where `c` is a natural number
* sampling: `X ~ <distribution>;` where `<distribution>` is `Bernoulli(p)`, `Categorical(p0, p1, ..., pn)` or `Geometric(p)`
* observations: `observe <event>` where `<event>` is described below. `fail` means observing an event that happens with probability 0.
* branching: `if <event> { <statements> } else { <statements> }` where `<event>` is described below
* looping: `while <event> { <statements> }`

**Distributions**: The following distributions are supported (where `m`, `n` are natural numbers, `a`, `b` are rational numbers and `p` is a rational number between 0 and 1).
Rational numbers can be written as decimals (e.g. `0.4`) or fractions (e.g. `355/113`).

* `Bernoulli(p)`
* `Categorical(p_0, p_1, ..., p_n)`: categorical distribution on `{0, ..., n}` where `i` has probability `p_i`
* `UniformDisc(a, b)`: uniform distribution on `{a, ..., b - 1}`
* `Geometric(p)`

**Events** take one of the following forms:

* `n ~ <distribution>`: the event of sampling `n` from the given distribution
* `flip(p)`: happens with probability `p` (short for `1 ~ Bernoulli(p)`)
* `X in [n1, n2, n3, ...]`: the event of `X` being in the given list of natural numbers
* `X not in [n1, n2, n3, ...]`
* `X = n`, `X != n`, `X < n`, `X <= n`, `X > n`, `X >= n` for a variable `X` and a natural number `n`
* `not <event>`: negation/complement
* `<event> and <event>`: conjunction/intersection
* `<event> or <event>`: disjunction/union

## 4.3 Organization of the source code

The source code is organized as follows:

- `benchmarks/`: directory containing the benchmarks and benchmark scripts
  - `ours/`: new benchmarks that we contributed (some are adapted from existing benchmarks)
  - `output/`: directory for GuBPI output files
  - `outputs/`: directory for output of our tool (for Fig. 7 in the paper)
  - `plots/`: plotting scripts (for Fig. 7)
  - `polar/`: benchmarks from the Polar repository
  - `prodigy/`: benchmarks from the Prodigy repository
  - `psi/`: benchmark from the PSI repository
- `dependencies/`: two Rust dependencies which had to be patched
- `src/`: source code of our implementation
  - `bin/`: binaries
    - `geobound.rs`: code for the `geobound` binary implementing the Geometric Bound Semantics
    - `residual.rs`: code for the `residual` binary implementing the Residual Mass Semantics
    - `stats.rs`: code for an auxiliary `stats` binary to report information about a probabilistic program (e.g. number of variables)
  - `bound/`: for data structures implementing bounds on distributions
    - `egd.rs`: implements EGDs (eventually geometric distributions)
    - `finite_discrete.rs`: implements finite discrete distributions
    - `geometric.rs`: implements a geometric bound, aggregating a finite discrete lower bound and an EGD upper bound
    - `residual.rs`: implements a residual mass bound, consisting of a finite discrete lower bound and the residual mass
  - `number/`: for number types
    - `f64.rs`: an extension of double-precision floating point numbers.
      Floating point numbers are used in some places to speed up the computations.
      But all results are verified with rational number computations.
    - `float_rat.rs`: for storing a rational number and its closest floating point number
    - `number.rs`: traits for number types
    - `rational.rs`: rational number type
  - `semantics/`: implementations of the transformer semantics from the paper
    - `geometric.rs`: the geometric bound semantics, which generates polynomial inequality constraints
    - `residual.rs`: the residual mass semantics
    - `support.rs`: for overapproximating the support of variables
  - `solvers/`: implementations of the solvers and optimizers for the polynomial inequality constraints
    - `adam.rs`: the `adam-barrier` solver that combines the barrier method with the popular ADAM algorithm
    - `ipopt.rs`: to run the IPOPT solver
    - `linear.rs`: to run an LP solver (COIN-CBC) to optimize the linear variables of the optimization problem
    - `problem.rs`: data structure for the constraint problem
    - `z3.rs`: to run the Z3 SMT solver
  - `interval.rs`: an interval data type
  - `parser.rs`: for parsing SGCL programs
  - `ppl.rs`: data structures for constructs in the PPL (probabilistic programming language)
  - `support.rs`: data structures for the support set of program variables
  - `sym_expr.rs`: data structures for symbolic expressions and constraints
  - `util.rs`: contains miscellaneous functions
  - `test.py`: a script to test changes during development (irrelevant for the artifact evaluation)
