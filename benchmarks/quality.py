#!/usr/bin/env python3

import json
import os
from pathlib import Path
import sys
from bench import *
import math

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

true_tail_re = re.compile(r"true tail: (.*)")
tail_bound_zero_re = re.compile(r"Asymptotics: p\(n\) = 0")
tail_bound_re = re.compile(r"Asymptotics: p\(n\) (?:(?:.*) \* ([e.0123456789+-]+)\^n)")
true_ev_re = re.compile(r"true EV: (.*)")
exact_ev_re = re.compile(r"1-th \(raw\) moment = ([e.0123456789+-]+)")
ev_bound_re = re.compile(r"1-th \(raw\) moment ∈ \[([eNa.0123456789+-]+), ([eNa.0123456789+-]+)\]")

if __name__ == "__main__":
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    with open("bench-results.json") as f:
        results = json.load(f)
    print(r"\begin{tabular}{l|ccc|ccc}")
    print(r"\toprule")
    print(r"Benchmark & True EV & EV bound & Time & True tail & Tail bound & Time \\")
    print(r"\midrule")
    for benchmark, bench_result in results.items():
        if not bench_result:
            continue
        program = Path(f"{benchmark}.sgcl").read_text()
        ev_result = bench_result["geobound-ev"]
        tail_result = bench_result["geobound-tail"]
        m = exact_ev_re.search(ev_result["stdout"])
        if m:
            ev_lo = ev_hi = float(m.group(1))
            ev_bound = f"{ev_lo:.6f}"
            true_ev = ev_bound
        else:
            ev_bound = r"\xmark{}"
            true_ev = "?"
        
        m = ev_bound_re.search(ev_result["stdout"])
        if m:
            ev_lo = round_down(float(m.group(1)), 6)
            ev_hi = round_up(float(m.group(2)), 6)
            ev_bound = f"[{ev_lo:.6g}, {ev_hi:.6g}]"
            true_ev = "?"
        else:
            ev_bound = r"\xmark{}"
            true_ev = "?"
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
        if m:
            tail_bound = float(m.group(1))
            tail_bound = f"$O({tail_bound:.4g}^n)$"
        elif tail_bound_zero_re.search(tail_result["stdout"]):
            tail_bound = r"$0$"
            true_tail = r"$0$"
        else:
            tail_bound = r"\xmark{}"
        m = true_tail_re.search(program)
        if m:
            true_tail = m.group(1).strip()
            true_tail = f"$0$" if true_tail == "0" else f"$\Theta({m.group(1)})$"
        tail_runtime = tail_result["time"]
        tail_runtime = f"{tail_runtime:.2f} s"
        tail_error = tail_result["error"]
        if tail_error == "timeout":
            tail_runtime = r"t/o"
        print(
                fr"\verb|{benchmark}| & {true_ev} & {ev_bound} & {ev_runtime} & {true_tail} & {tail_bound} & {tail_runtime} \\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")
