#!/usr/bin/env python3

# This script analyzes the file `bench-results.json` produced by `bench.py`.
# It prints a LaTeX table with the results.

import json
import os
from pathlib import Path
import sys


tool_groups = [["genfer", "dice"], ["genfer-rational", "dice-rational", "prodigy", "psi"]]
tool_names = {
    "genfer": "Genfer (FP)",
    "genfer-rational": "Genfer ($\mathbb{Q}$)",
    "prodigy": "Prodigy",
    "dice": "Dice (FP)",
    "dice-rational": "Dice ($\mathbb{Q}$)",
    "psi": "PSI",
}

def s(time):
    if time < 0.01:
        return f"{time:.4f}"
    if time < 0.1:
        return f"{time:.3f}"
    if time < 1:
        return f"{time:.2f}"
    if time < 10:
        return f"{time:.1f}"
    return f"{time:.0f}"

def latex_table(all_results):
    benchmarks = list(all_results.keys())
    benchmarks.sort()
    cols = "l"
    for tool_group in tool_groups:
        cols += "|"
        for tool in tool_group:
            cols += "r"
    print(fr"\begin{{tabular}}{{{cols}}}")
    print("Tool", end=" ")
    for tool_group in tool_groups:
        for tool in tool_group:
            print(f"&{tool_names[tool]}", end=" ")
    first = True
    for benchmark in benchmarks:
        print(r"\\")
        if first:
            print(r"\hline")
            first = False
        result = all_results[benchmark]
        print(benchmark, end=" ")
        for tool_group in tool_groups:
            minimum = None
            for time in [result[tool] for tool in tool_group]:
                if not isinstance(minimum, float) or (isinstance(time, float) and time < minimum):
                    minimum = time
            for tool in tool_group:
                time = result[tool]
                highlight = r"\textbf" if time == minimum else ""
                if isinstance(time, float):
                    print(f"&{highlight}{{{s(time)}s}}", end=" ")
                else:
                    print(f"&{time}", end=" ")
    print()
    print(r"\end{tabular}")

if __name__ == "__main__":
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    with open("bench-results.json", "r") as f:
        all_results = json.load(f)
    assert isinstance(all_results, dict)
    benchmarks = all_results.keys()
    print("LATEX OUTPUT:")
    latex_table(all_results)
