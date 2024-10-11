from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

blue = "\033[94m"
green = "\033[92m"
red = "\033[91m"
bold = "\033[1m"
reset = "\033[0m"

timeout = 10 # seconds
default_unroll = 30

total_time_re = re.compile(r"(?:Total|Elapsed) time: ([0-9.]*) *s")
tail_bound_zero_re = re.compile(r"Asymptotics: p\(n\) = 0")
tail_bound_re = re.compile(r"Asymptotics: p\(n\) (?:(?:.*) \* ([e.0123456789+-]+)\^n)")
exact_ev_re = re.compile(r"1-th \(raw\) moment = ([e.0123456789+-]+)")
ev_bound_re = re.compile(r"1-th \(raw\) moment ∈ \[([e.0123456789+-]+), ([e.0123456789+-]+)\]")

def bench(benchmark, unroll=None, inv_size=None, flags=None):
    benchmark = { "name": benchmark }
    if unroll is not None:
        benchmark["unroll"] = unroll
    if inv_size is not None:
        benchmark["inv_size"] = inv_size
    if flags is not None:
        benchmark["flags"] = flags
    return benchmark

benchmarks = [
    "ours/1d-asym-rw",
    bench("ours/2d-asym-rw", unroll=20, inv_size=2),
    # "ours/3d-asym-rw",
    "ours/asym-rw-conditioning",
    bench("ours/coupon-collector5", unroll=5, flags=["--optimizer", "ipopt"]),
    "ours/double-geo",
    "ours/geometric",
    bench("ours/grid", unroll=20, inv_size=2),
    # "ours/herman5",
    bench("ours/imprecise_tails", unroll=20, inv_size=2),
    # "ours/israeli-jalfon4",
    bench("ours/nested", unroll=3, inv_size=2),
    "ours/sub-geom",
    "ours/sum-geos",
]

def compile():
    cargo_command = "cargo build --release --bin residual --bin geobound"
    print(f"Running `{cargo_command}`...")
    with subprocess.Popen(
        cargo_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        for line in process.stdout:
            print(line.decode("utf8"))

def run_benchmark(benchmark, flags):
    command = ["target/release/geobound", f"benchmarks/{benchmark}.sgcl"] + flags
    try:
        env = os.environ.copy()
        env["RUST_BACKTRACE"] = "1"
        completed = subprocess.run(
                command, timeout=timeout, capture_output=True, env=env
            )
        stdout = (completed.stdout or b"").decode("utf-8")
        stderr = (completed.stderr or b"").decode("utf-8")
        exitcode = completed.returncode
        if exitcode != 0:
            print(f"Error running {benchmark}. Exit code: {exitcode}")
            print("Command: ", " ".join(command))
            print("Error:")
            print(stderr)
            return "crash"
        return stdout
    except subprocess.TimeoutExpired:
        print(f"Timeout for {benchmark}")
        print("Command: ", " ".join(command))
        return "timeout"

def run_benchmarks():
    compile()
    start = time.time()
    results = {}
    for benchmark in benchmarks:
        if isinstance(benchmark, str):
            benchmark = bench(benchmark)
        name = benchmark["name"]
        unroll = benchmark.get("unroll", default_unroll)
        inv_size = benchmark.get("inv_size", 1)
        flags = benchmark.get("flags", [])
        print(f"Running benchmark {name}...")
        result = {}
        out = run_benchmark(name, ["--objective", "ev", "-u", f"{unroll}", "-d", f"{inv_size}"] + flags)
        if out == "crash" or out == "timeout":
            result["ev"] = out
            result["time_ev"] = timeout
        else:
            m = exact_ev_re.search(out)
            if m:
                result["ev"] = float(m.group(1))
            m = ev_bound_re.search(out)
            if m:
                result["ev"] = float(m.group(2))
            m = total_time_re.search(out)
            if m:
                result["time_ev"] = float(m.group(1))
        out = run_benchmark(name, ["--objective", "tail", "-u", "0", "-d", f"{inv_size}"] + flags)
        if out == "crash" or out == "timeout":
            result["tail"] = out
            result["time_tail"] = timeout
        else:
            m = tail_bound_re.search(out)
            if m:
                result["tail"] = float(m.group(1))
            elif tail_bound_zero_re.search(out):
                result["tail"] = 0
            m = total_time_re.search(out)
            if m:
                result["time_tail"] = float(m.group(1))
        results[name] = result
    end = time.time()
    print(f"{blue}Total time: {bold}{end - start:.2f} s{reset}\n")
    return results

def compare_measurements(key, base, test, eq_tol=1.05, small_tol=1.25):
    if isinstance(base, str):
        if isinstance(test, str):
            format = ""
            comment = "both failed"
            result = 0
        else:
            format = green + bold
            comment = "fixed"
            result = 2
    elif isinstance(test, str):
        format = red + bold
        comment = "regressed"
        result = -2
    elif base <= test * eq_tol and test <= base * eq_tol:
        format = ""
        comment = "similar"
        result = 0
    elif test > base and test <= base * small_tol:
        format = red
        comment = "slightly worse"
        result = -1
    elif test < base and base <= test * small_tol:
        format = green
        comment = "slightly better"
        result = 1
    elif test > base:
        format = red + bold
        comment = "worse"
        result = -2
    else:
        format = green + bold
        comment = "better"
        result = 2
    if isinstance(base, float):
        base = f"{base:.4g}"
    if isinstance(test, float):
        test = f"{test:.4g}"
    print(f"  {key:>10}:  {format}{base:>9} -> {test:>9}  ({comment}){reset}")
    return result

def compare_time(key, baseline, test):
    base = baseline[key]
    test = test[key]
    return compare_measurements(key, base, test)

def compare_metric(key, baseline, test):
    base = baseline[key]
    test = test[key]
    return compare_measurements(key, base, test)

def compare_benchmark(key, baseline, test):
    result = {}
    print(f"Benchmark {key}:")
    result["time_ev"] = compare_time("time_ev", baseline, test)
    result["ev"] = compare_metric("ev", baseline, test)
    result["time_tail"] = compare_time("time_tail", baseline, test)
    result["tail"] = compare_metric("tail", baseline, test)
    return result

def compare():
    with open("baseline.json") as f:
        baseline = json.load(f)
    with open("test.json") as f:
        test = json.load(f)
    keys = set(baseline.keys()) | set(test.keys())
    total = defaultdict(int)
    for key in sorted(keys):
        if key not in baseline:
            print(f"Test result for {key} is missing in baseline")
        elif key not in test:
            print(f"Baseline result for {key} is missing in test")
        else:
            result = compare_benchmark(key, baseline[key], test[key])
            for key, count in result.items():
                total[key] += count
    for key, count in total.items():
        if count == 0:
            print(f"{key}: {blue}similar{reset}")
        elif count > 0:
            print(f"{key}: {green}{count} better{reset}")
        elif count < 0:
            print(f"{key}: {red}{-count} worse{reset}")

if __name__ == '__main__':
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    if len(sys.argv) < 2:
        print("Usage: test.py baseline|test|compare")
        sys.exit(1)
    if sys.argv[1] == "baseline":
        results = run_benchmarks()
        with open("baseline.json", "w") as f:
            json.dump(results, f, indent=2)
    elif sys.argv[1] == "test":
        results = run_benchmarks()
        with open("test.json", "w") as f:
            json.dump(results, f, indent=2)
        compare()
    elif sys.argv[1] == "compare":
        compare()