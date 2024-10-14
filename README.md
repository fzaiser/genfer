# Artifact for: "Guaranteed Bounds on Posterior Distributions of Discrete Probabilistic Programs with Loops"

## Abstract

TODO



## 1 Instructions for Running the Virtual Machine

### 1.1 Download

* Download the README file `README.md` and the VM image `popl2025.ova` from TODO.
* Download Oracle VirtualBox 7.1.2 for your system here: https://www.virtualbox.org/wiki/Downloads.
* Follow the instructions in `README.md`.



### 1.2 Setup

Import the VM image `popl2025.ova` and start the virtual machine.
The guest OS is Ubuntu 22.04.05.
Log in with user name `user` and password `123`.
All dependencies are already installed on the VM.

Make sure that the keyboard layout "en" (English US), not "de", is selected (top right of the screen, next to the network indicator etc.).

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





## 2 Kicking the Tires

### 2.1 Trying out the tool

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



## 2.2 Reproducing the plots from the paper (Fig. 7)

Next, let's reproduce the plots from Fig. 7 in the paper.
To do this, run the following commands:

```shell
cd tool/benchmarks # if not already
# Generate the data for the plots (takes about 1min):
./comparison.sh
# Plot the data:
./plots.sh
# Now the plots (named `plot_benchmark.pdf`) are in the `benchmarks/` directory
```

The script `./comparison.sh` runs the Geometric Bound Semantics on 5 benchmarks: `asym_rw.sgcl`, `coupon-collector.sgcl`, `die_paradox.sgcl`, `geo.sgcl`, `herman.sgcl`.
Each benchmark is run 4 times: once with the Residual Mass Semantics and three times with the Geometric Bound Semantics, each time with a different optimization objective (`total` for probability masses, `ev` for moments, or `tail` for tail asymptotics).






## Probabilistic programming language

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
