#!/usr/bin/env python3

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from bench import *


class ProgramStats:
    def __init__(self, var_count, stmt_count, support, has_observations):
        self.var_count = var_count
        self.stmt_count = stmt_count
        self.support = support
        self.has_observations = has_observations


stat_tool_path = env("STAT_TOOL", "../../target/debug/stats")

# e.g. "12 variables"
var_count_re = re.compile(r"([0-9]+) variables")
# e.g. "12 statements"
stmt_count_re = re.compile(r"([0-9]+) statements")
# e.g. "Support: {0}, {2, ..., 4}, {1, ...}"
supports_re = re.compile(r"Support: (.*)")
# e.g. "{0}", or "{2, ..., 4}", or "{1, ...}"
support_re = re.compile(r"\{[^}]*\}")

point_re = re.compile(r"\{([0-9]+)\}")
fin_range_re = re.compile(r"\{([0-9]+), \.\.\., ([0-9]+)\}")
inf_range_re = re.compile(r"\{([0-9]+), \.\.\.\}")

constr_count_re = re.compile(r"Generated ([0-9]+) constraints")
constr_var_count_re = re.compile(r"with ([0-9]+) symbolic variables")


def cardinality(supset):
    if point_re.fullmatch(supset):
        return 1
    elif fin_range_re.fullmatch(supset):
        start = int(fin_range_re.search(supset).group(1))
        end = int(fin_range_re.search(supset).group(2))
        return end - start + 1
    elif inf_range_re.fullmatch(supset):
        return float("inf")


def format_support(support):
    if support == float("inf"):
        return r"\infty"
    else:
        return str(support)


def program_stats(path: Path):
    command = [stat_tool_path, str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    out = result.stdout
    var_count = var_count_re.search(out).group(1)
    stmt_count = stmt_count_re.search(out).group(1)
    supports = supports_re.search(out).group(1)
    support = [cardinality(supset) for supset in support_re.findall(supports)]
    has_observations = "Contains observations: true" in out
    return ProgramStats(var_count, stmt_count, support, has_observations)


if __name__ == "__main__":
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    with open("bench-results.json") as f:
        results = json.load(f)
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"Benchmark & \#vars & \#stmts & obs.? & \#constrs & \#constr vars & Time \\")
    print(r"\midrule")
    for benchmark, bench_result in results.items():
        for tool, result in bench_result.items():
            stats = program_stats(result["path"])
            stdout = result["stdout"]
            if tool != "geobound":
                continue
            tool = "Geom. bound"
            m = constr_count_re.search(stdout)
            if m:
                constr_count = m.group(1)
            else:
                constr_count = "n/a"
            m = constr_var_count_re.search(stdout)
            if m:
                constr_var_count = constr_var_count_re.search(stdout).group(1)
            else:
                constr_var_count = "n/a"
            obs = r"\cmark" if stats.has_observations else r"\xmark"
            support = r" \times ".join([format_support(sup) for sup in stats.support])
            runtime = result["time"]
            runtime = f"{runtime:.2f} s"
            error = result["error"]
            if error:
                if error == "timeout" and "rounding error" in stdout:
                    runtime = f"\\xmark{{}} (rounding errors)"
                elif error == "timeout":
                    runtime = f"\\xmark{{}} (> {runtime})"
                else:
                    runtime = f"\\xmark{{}}"
            print(
                f"\\verb|{benchmark}| & {stats.var_count} & {stats.stmt_count} & {obs} & {constr_count} & {constr_var_count} & {runtime} \\\\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")
