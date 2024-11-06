#!/usr/bin/env python3

import json
import os
from pathlib import Path
import re
import sys
import subprocess
import math

# e.g. "true tail: 0.5^n":
true_tail_re = re.compile(r"true tail: (.*)")
# e.g. "Asymptotics: p(n) = 0":
tail_bound_zero_re = re.compile(r"Asymptotics: p\(n\) = 0")
# e.g. "Asymptotics: p(n) = 10 * 0.5^n":
tail_bound_re = re.compile(r"Asymptotics: p\(n\) (?:(?:.*) \* ([e.0123456789+-]+)\^n)")
# e.g. "true EV: 0.5":
true_ev_re = re.compile(r"true EV: (.*)")
# e.g. "1-th (raw) moment = 0.5":
exact_ev_re = re.compile(r"1-th \(raw\) moment = ([Nae.0123456789+-]+)")
# e.g. "2-th (raw) moment = 0.5":
exact_mom2_re = re.compile(r"2-th \(raw\) moment = ([Nae.0123456789+-]+)")
# e.g. "1-th (raw) moment ∈ [0.5, 0.6]":
ev_bound_re = re.compile(r"1-th \(raw\) moment ∈ \[([Nae.0123456789+-]+), ([Nainfe.0123456789+-]+)\]")
# e.g. "2-th (raw) moment ∈ [0.5, 0.6]":
mom2_bound_re = re.compile(r"2-th \(raw\) moment ∈ \[([Nae.0123456789+-]+), ([Nainfe.0123456789+-]+)\]")
# e.g. "E[var] = 0.5":
polar_ev_re = re.compile(r"E\([A-Za-z0-9_]*\) = (.*)")
# e.g. "Total time: 42.0 s":
total_time_re = re.compile(r"(?:Total|Elapsed) time: ([0-9.]*) *s")

# e.g. "12 variables"
var_count_re = re.compile(r"([0-9]+) variables")
# e.g. "12 statements"
stmt_count_re = re.compile(r"([0-9]+) statements")
# e.g. "Support: {0}, {2, ..., 4}, {1, ...}"
supports_re = re.compile(r"Support: (.*)")
# e.g. "{0}", or "{2, ..., 4}", or "{1, ...}"
support_re = re.compile(r"\{[^}]*\}")
# e.g. "{42}":
point_re = re.compile(r"\{([0-9]+)\}")
# e.g. "{2, ..., 4}":
fin_range_re = re.compile(r"\{([0-9]+), \.\.\., ([0-9]+)\}")
# e.g. "{1, ...}":
inf_range_re = re.compile(r"\{([0-9]+), \.\.\.\}")

# e.g. "Generated 42 constraints":
constr_count_re = re.compile(r"Generated ([0-9]+) constraints")
# e.g. "with 42 symbolic variables":
constr_var_count_re = re.compile(r"with ([0-9]+) symbolic variables")

stat_tool_path = "../target/release/stats"

def round_down(x, significant_figures):
    if not math.isfinite(x) or x == 0:
        return x
    int_digits = math.floor(math.log10(x))
    shift = significant_figures - int_digits - 1
    return math.floor(x * 10**shift) / 10**shift

def round_up(x, significant_figures):
    if not math.isfinite(x) or x == 0:
        return x
    int_digits = math.floor(math.log10(x))
    shift = significant_figures - int_digits - 1
    return math.ceil(x * 10**shift) / 10**shift

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

class ProgramStats:
    def __init__(self, var_count, stmt_count, support, has_observations):
        self.var_count = var_count
        self.stmt_count = stmt_count
        self.support = support
        self.has_observations = has_observations


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

def extract_ev(output):
    m = exact_ev_re.search(output)
    ev_lo = ev_hi = None
    if m:
        ev_lo = ev_hi = float(m.group(1))
    m = ev_bound_re.search(output)
    if m:
        ev_lo = float(m.group(1))
        ev_hi = float(m.group(2))
    return ev_lo, ev_hi

def extract_mom2(output):
    m = exact_mom2_re.search(output)
    mom2_lo = mom2_hi = None
    if m:
        mom2_lo = mom2_hi = float(m.group(1))
    m = mom2_bound_re.search(output)
    if m:
        mom2_lo = float(m.group(1))
        mom2_hi = float(m.group(2))
    return mom2_lo, mom2_hi

def extract_decay_rate(output):
    m = tail_bound_re.search(output)
    if m:
        return float(m.group(1))
    return None


def extract_time(output):
    m = total_time_re.search(output)
    if m:
        return float(m.group(1))
    return None


def applicability_table():
    with open("bench-results.json") as f:
        results = json.load(f)
    print(r"========================================")
    print(r"APPLICABILITY TABLE:")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"Benchmark & \#V & \#S & O? & \#C & \#CV & Time \\")
    print(r"\midrule")
    for benchmark, bench_result in results.items():
            result = bench_result["geobound-existence"]
            stats = program_stats(result["path"])
            stdout = result["stdout"]
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


def quality_of_bounds_table():
    with open("bench-results.json") as f:
        results = json.load(f)
    print(r"========================================")
    print(r"QUALITY OF BOUNDS TABLE:")
    print(r"\begin{tabular}{l|cccc|ccc}")
    print(r"\toprule")
    print(r"Benchmark & \#U & True EV & EV bound & Time & True tail & Tail bound & Time \\")
    print(r"\midrule")
    for benchmark, bench_result in results.items():
        if not bench_result:
            continue
        if bench_result["geobound-existence"]["error"]:
            continue
        ev_flags = bench_result["geobound-ev"]["flags"]
        ev_unroll = ev_flags[ev_flags.index("-u") + 1]
        program = Path(f"{benchmark}.sgcl").read_text()
        ev_result = bench_result["geobound-ev"]
        tail_result = bench_result["geobound-tail"]

        m = constr_count_re.search(ev_result["stdout"])
        if m:
            constr_count = m.group(1)
        else:
            constr_count = "n/a"
        m = constr_var_count_re.search(ev_result["stdout"])
        if m:
            constr_var_count = constr_var_count_re.search(ev_result["stdout"]).group(1)
        else:
            constr_var_count = "n/a"
        m = exact_ev_re.search(ev_result["stdout"])
        ev_bound = r"\xmark{}"
        true_ev = "?"
        if m:
            ev_lo = ev_hi = float(m.group(1))
            ev_bound = f"{ev_lo:.4f}"
            true_ev = ev_bound
        m = ev_bound_re.search(ev_result["stdout"])
        if m:
            ev_lo = round_down(float(m.group(1)), 4)
            ev_hi = round_up(float(m.group(2)), 4)
            ev_bound = f"[{ev_lo:.4g}, {ev_hi:.4g}]"
        m = true_ev_re.search(program)
        if m:
            true_ev = m.group(1)
        ev_runtime = ev_result["time"]
        ev_runtime = f"{ev_runtime:.2f} s"
        ev_error = ev_result["error"]
        if ev_error == "timeout":
            ev_runtime = r"t/o"

        true_tail = "?"
        m = tail_bound_re.search(tail_result["stdout"])
        tail_bound = r"\xmark{}"
        if m:
            tail_bound = round_up(float(m.group(1)), 4)
            tail_bound = f"$O({tail_bound:.4g}^n)$"
        elif tail_bound_zero_re.search(tail_result["stdout"]):
            tail_bound = r"$0$"
            true_tail = r"$0$"
        m = true_tail_re.search(program)
        if m:
            true_tail = m.group(1).strip()
            true_tail = f"$0$" if true_tail == "0" else f"$\Theta({true_tail})$"
        tail_runtime = tail_result["time"]
        tail_runtime = f"{tail_runtime:.2f} s"
        tail_error = tail_result["error"]
        if tail_error == "timeout":
            tail_runtime = r"t/o"
        print(
                fr"\verb|{benchmark}| & {ev_unroll} & {true_ev} & {ev_bound} & {ev_runtime} & {true_tail} & {tail_bound} & {tail_runtime} \\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def polar_comparison_table():
    with open("bench-results.json") as f:
        results = json.load(f)
    print(r"========================================")
    print(r"POLAR COMPARISON TABLE:")
    print(r"\begin{tabular}{l|cc|cc}")
    print(r"\toprule")
    print(r"Benchmark & exact EV (Polar) & time (Polar) & EV bound (ours) & time (ours) \\")
    print(r"\midrule")
    for benchmark, bench_result in results.items():
        if not bench_result or "geobound-ev" not in bench_result or "polar" not in bench_result:
            continue
        if bench_result["geobound-existence"]["error"]:
            continue

        ev_result = bench_result["geobound-ev"]
        m = exact_ev_re.search(ev_result["stdout"])
        ev_bound = r"\xmark{}"
        if m:
            ev_lo = ev_hi = float(m.group(1))
            ev_bound = f"{ev_lo:.4f}"
        m = ev_bound_re.search(ev_result["stdout"])
        if m:
            ev_lo = round_down(float(m.group(1)), 4)
            ev_hi = round_up(float(m.group(2)), 4)
            ev_bound = f"[{ev_lo:.4g}, {ev_hi:.4g}]"
        ev_runtime = ev_result["time"]
        ev_runtime = f"{ev_runtime:.2f} s"
        ev_error = ev_result["error"]
        if ev_error == "timeout":
            ev_runtime = r"t/o"

        polar_result = bench_result["polar"]
        m = polar_ev_re.search(polar_result["stdout"])
        if m:
            polar_ev = m.group(1)
        else:
            polar_ev = r"\xmark{}"
        polar_runtime = polar_result["time"]
        polar_runtime = f"{polar_runtime:.2f} s"
        polar_error = polar_result["error"]
        if polar_error == "timeout":
            polar_runtime = r"t/o"

        print(
                fr"\verb|{benchmark}| & {polar_ev} & {polar_runtime} & {ev_bound} & {ev_runtime} \\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def semantics_comparison_table():
    print(r"========================================")
    print(r"TABLE FOR COMPARISON BETWEEN THE TWO SEMANTICS:")
    benchmarks = ["geo", "asym_rw", "die_paradox", "coupon_collector", "herman"]
    exact = {
        "geo": [1, 3, "$\Theta(0.5^n)$"],
        "asym_rw": [2, 10, "$\Theta(n^{-3/2}0.8660...^n)$"],
        "die_paradox": [1.5, 3, "$\Theta(0.3333...^n)$"],
        "coupon_collector": ["11.41666...", "155.513888...", "$\Theta(0.8^n)$"],
        "herman": ["1.333...", "4.222...", "?"]
    }
    print(r"\begin{tabular}{l|lllll}")
    print(r"\toprule")
    print(r"Example &Method &Expected value &2nd moment &Tail &Time \\")
    for benchmark in benchmarks:
        print(r"\midrule")
        exact_ev, exact_mom2, exact_tail = exact[benchmark]
        print(fr"\verb|{benchmark}| &exact &{exact_ev} &{exact_mom2} &{exact_tail} & \\")
        residual_output = Path(f"outputs/{benchmark}_residual.txt").read_text()
        ev_lo, _ = extract_ev(residual_output)
        mom2_lo, _ = extract_mom2(residual_output)
        geometric_moments_output = Path(f"outputs/{benchmark}_bound_moments.txt").read_text()
        _, ev_hi = extract_ev(geometric_moments_output)
        _, mom2_hi = extract_mom2(geometric_moments_output)
        geometric_tail_output = Path(f"outputs/{benchmark}_bound_tail.txt").read_text()
        decay = extract_decay_rate(geometric_tail_output)
        residual_time = extract_time(residual_output)
        geometric_time = max(extract_time(geometric_moments_output), extract_time(geometric_tail_output))
        print(fr" &residual &$\ge {ev_lo}$ &$\ge {mom2_lo}$ &n/a &{residual_time:.3f} s \\")
        print(fr" &geometric &$\le {ev_hi}$ &$\le {mom2_hi}$ &$O({round_up(decay, 4):.4f}^n)$ &{geometric_time:.3f} s \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    if len(sys.argv) == 1 or "applicability" in sys.argv:
        applicability_table()
    if len(sys.argv) == 1 or "quality" in sys.argv:
        quality_of_bounds_table()
    if len(sys.argv) == 1 or "polar-comparison" in sys.argv:
        polar_comparison_table()
    if len(sys.argv) == 1 or "semantics-comparison" in sys.argv:
        semantics_comparison_table()
