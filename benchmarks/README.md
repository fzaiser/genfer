# Reproducing the experiments

## Applicability of the geometric bound semantics (Section 6.1)

* The benchmarks can be found in `own/`, `polar/`, `prodigy/` and `psi/`.
  Each folder contains a file `unsupported.txt` with a list of benchmarks that we filtered out and for what reason (e.g. it uses continuous distributions).
* The script `bench.py` runs our tool on all of them and collects the outputs in `bench-results.json`.
* The script `existence.py` reads this file and outputs the data for Table 2.

## Comparison between our two semantics (Section 6.3)

* The three benchmark programs are available in the files `geometric.sgcl`, `asym_rw.sgcl`, and `die_paradox.sgcl`.
* The script `comparison.sh` runs our tool with various settings on the three examples.
  We pick different optimization objectives depending on the purpose: for the moment bounds, the objective is the expected value bound; for tails, it is the tail decay rate; for probability masses, it is the total probability mass bound.
* The output is written to `outputs/`.
  The contents in Table 3 are taken from `outputs/<benchmark>_residual.txt` for the residual mass semantics; and for the geometric bound semantic, from `outputs/<benchmark>_moments.txt` (optimized for the expected value) for the moments and `outputs/<benchmark>_tail.txt` (optimized for the tails) for the tail asymptotics.
* The scripts in `plots/plot_<benchmark>.py` are used to create Fig. 8 in the paper.
  They read the outputs `outputs/<benchmarks>_residual.txt` for the residual mass semantics and `outputs/<benchmarks>_bound_probs.txt` (optimized for)
