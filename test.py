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
exact_ev_re = re.compile(r"1-th \(raw\) moment = ([Nae.0123456789+-]+)")
ev_bound_re = re.compile(r"1-th \(raw\) moment ∈ \[([Nae.0123456789+-]+), ([Nae.0123456789+-]+)\]")

def bench(benchmark, unroll=None, inv_size=None, flags=None, slow=False):
    benchmark = { "name": benchmark }
    if unroll is not None:
        benchmark["unroll"] = unroll
    if inv_size is not None:
        benchmark["inv_size"] = inv_size
    if flags is not None:
        benchmark["flags"] = flags
    benchmark["slow"] = slow
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
    bench("polar/c4B_t303", inv_size=2),
    "polar/coupon_collector2",
    "polar/fair_biased_coin",
    # "polar/geometric",
    "polar/las_vegas_search",
    "polar/linear01",
    # "polar/rabin",
    # "polar/random_walk_2d",
    "polar/simple_loop",
    "prodigy/bit_flip_conditioning",
    "prodigy/brp_obs",
    "prodigy/condand",
    "prodigy/dep_bern",
    "prodigy/endless_conditioning",
    "prodigy/geometric",
    "prodigy/ky_die",
    "prodigy/n_geometric",
    # "prodigy/nested_while",
    # "prodigy/random_walk",
    "prodigy/trivial_iid",
    bench("psi/beauquier-etal3", slow=True),
    "psi/cav-example7",
    "psi/dieCond",
    "psi/ex3",
    "psi/ex4",
    bench("psi/fourcards", slow=True),
    bench("psi/herman3", slow=True),
    bench("psi/israeli-jalfon3", slow=True),
    # "psi/israeli-jalfon5",
]

class BenchRun:
    def __init__(self, command, stdout, stderr, exitcode, timeout, error=None):
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.exitcode = exitcode
        self.timeout = timeout
        self.error = error

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
    print(f"$ {' '.join(command)}")
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
            print(f"  {red}crashed (exitcode {exitcode}){reset}")
            print(f"  STDERR:")
            print("  " + stderr.replace("\n", "\n  "))
            result = BenchRun(command, stdout, stderr, exitcode, timeout, "crash")
        else:
            result = BenchRun(command, stdout, stderr, exitcode, timeout)
    except subprocess.TimeoutExpired as e:
        print(f"  {red}timeout{reset}")
        stdout = (e.stdout or b"").decode("utf-8")
        stderr = (e.stderr or b"").decode("utf-8")
        result = BenchRun(command, stdout, stderr, None, timeout, "timeout")
    return result

def run_benchmarks(run_slow=False, out_file=None):
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
        slow = benchmark.get("slow", False)
        if slow and not run_slow:
            print(f"Skipping slow benchmark {name}")
            continue
        result = {}
        run = run_benchmark(name, ["--objective", "ev", "-u", f"{unroll}", "-d", f"{inv_size}"] + flags)
        if run.error == "crash" or run.error == "timeout":
            result["ev"] = run.error
            result["time_ev"] = timeout
        else:
            m = exact_ev_re.search(run.stdout)
            if m:
                result["ev"] = float(m.group(1))
            m = ev_bound_re.search(run.stdout)
            if m:
                result["ev"] = float(m.group(2))
            m = total_time_re.search(run.stdout)
            if m:
                result["time_ev"] = float(m.group(1))
        if out_file:
            out_file.write(f"$ {' '.join(run.command)}\n")
            out_file.write(run.stdout)
            out_file.write("\n\n")
            out_file.write(run.stderr)
            out_file.write("\n\n\n\n")
        run = run_benchmark(name, ["--objective", "tail", "-u", "0", "-d", f"{inv_size}"] + flags)
        if run.error == "crash" or run.error == "timeout":
            result["tail"] = run.error
            result["time_tail"] = timeout
        else:
            m = tail_bound_re.search(run.stdout)
            if m:
                result["tail"] = float(m.group(1))
            elif tail_bound_zero_re.search(run.stdout):
                result["tail"] = 0
            m = total_time_re.search(run.stdout)
            if m:
                result["time_tail"] = float(m.group(1))
        if out_file:
            out_file.write(f"$ {' '.join(run.command)}\n")
            out_file.write(run.stdout)
            out_file.write("\n\n")
            out_file.write(run.stderr)
            out_file.write("\n\n\n\n")
        results[name] = result
    end = time.time()
    print(f"{blue}Total time: {bold}{end - start:.2f} s{reset}\n")
    return results

def compare_measurements(key, base, test, eq_tol=1.001, small_tol=1.25):
    if isinstance(base, str):
        if isinstance(test, str):
            format = ""
            comment = "both failed"
            result = 0
        else:
            format = green + bold
            comment = "fixed"
            result = 5
    elif isinstance(test, str):
        format = red + bold
        comment = "regressed"
        result = -5
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
    return compare_measurements(key, base, test, eq_tol=1.1)

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
            print(f"{blue}Test result for {key} is {red}missing in baseline{reset}")
        elif key not in test:
            print(f"{blue}Baseline result for {key} is {red}missing in test{reset}")
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
    run_slow = len(sys.argv) > 2 and sys.argv[2] == "slow"
    if sys.argv[1] == "baseline":
        with open("baseline.out", "w") as out_file:
            results = run_benchmarks(run_slow=run_slow, out_file=out_file)
        with open("baseline.json", "w") as f:
            json.dump(results, f, indent=2)
    elif sys.argv[1] == "test":
        with open("test.out", "w") as out_file:
            results = run_benchmarks(run_slow=run_slow, out_file=out_file)
        with open("test.json", "w") as f:
            json.dump(results, f, indent=2)
        compare()
    elif sys.argv[1] == "compare":
        compare()