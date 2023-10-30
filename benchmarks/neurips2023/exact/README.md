# Comparison with exact inference

## Benchmarks

These examples were taken from the following sources:

* https://github.com/eth-sri/psi/tree/9db68ba9581b7a1211f1514e44e7927af24bd398/test/r2
* https://github.com/eth-sri/psi/tree/9db68ba9581b7a1211f1514e44e7927af24bd398/test/fun
* https://github.com/SHoltzen/dice/tree/ed8671689a2a6466c8aaaee57dbc3e3b71150825/benchmarks/baselines

## Reproducing the experimental results

### Download and patch the exact inference tools

1. Clone the repositories for the tools Dice (https://github.com/SHoltzen/dice/, commit ed8671689a2a6466c8aaaee57dbc3e3b71150825), Prodigy (https://github.com/LKlinke/Prodigy/, commit 20c9d33c7b6bc8f4aca81e0e94905ba23d6c8558), and PSI (https://github.com/eth-sri/psi, commit 9db68ba9581b7a1211f1514e44e7927af24bd398).
2. Apply the patches in `tool-patches/`. They change the tool to measure and output the time taken for inference, excluding startup and parsing.
3. Build each tool according to its README.

### Running the tools

Set the environment variables `DICE` to the dice binary, `GENFER` to the Genfer binary `../genfer/target/release/genfer`, `PRODIGY` to the Prodigy repository folder, and `PSI` to the PSI binary.

Then run `python3 bench.py`, which will write the results to `bench-results.json`.
It may take a few hours to complete, but we include this JSON file already, so you can skip this step.
