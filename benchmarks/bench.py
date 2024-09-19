#!/usr/bin/env python3

from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

green = "\033[92m"
red = "\033[91m"
reset = "\033[0m"

benchmark_dirs = [
    "polar",
    "prodigy",
    "psi",
    "own",
]

timeout = 10  # TODO: increase
num_runs = 1  # TODO: increase

total_time_re = re.compile("Total time: ([0-9.]*)s")
flags_re = re.compile("flags: (.*)")
tool_flags_re = re.compile("flags ?\((.*)\): (.*)")
evbound_re = re.compile(r"1-th \(raw\) moment (.*)")
tailbound_re = re.compile(r"Asymptotics: p\(n\) (.*)")


def env(name, default):
    if name in os.environ:
        return os.environ[name]
    else:
        print(
            f"{red}Environment variable `{name}` is not set!{reset} Defaulting to `{default}`"
        )
        return default


class Tool:
    def __init__(self, name, path, flags=[]):
        self.name = name
        self.path = path
        self.flags = flags


residual_path = "../target/debug/residual" # TODO: Change to release
geo_bound_path = "../target/debug/geobound"  # TODO: Change to release
tools = [
    Tool("geobound-existence", geo_bound_path, ["-u", "0", "--keep-while", "--objective", "balance"]),
    Tool("geobound-ev", geo_bound_path, ["-u", "50", "--keep-while", "--objective", "ev"]),
    Tool("geobound-tail", geo_bound_path, ["-u", "0", "--keep-while", "--objective", "tail"]),
]


class BenchmarkResult:
    def __init__(
        self,
        tool,
        path,
        flags,
        time=None,
        stdout="",
        stderr="",
        error=None,
        exitcode=None,
        timeout=None,
        tailbound=None,
        evbound=None,
    ):
        self.tool = tool
        self.path = path
        self.flags = flags
        self.stdout = stdout
        self.stderr = stderr
        self.time = time
        self.error = error
        self.exitcode = exitcode
        self.timeout = timeout
        self.tailbound = tailbound
        self.evbound = evbound


def run_tool(tool, tool_command, path, flags, timeout):
    if not isinstance(tool_command, list):
        tool_command = [tool_command]
    try:
        command = tool_command + flags + [path]
        print(f"Running {command}...")
        start = time.perf_counter()
        env = os.environ.copy()
        env["RUST_BACKTRACE"] = "1"
        completed = subprocess.run(
            command, timeout=timeout, capture_output=True, env=env
        )
        elapsed = time.perf_counter() - start
        stdout = (completed.stdout or b"").decode("utf-8")
        stderr = (completed.stderr or b"").decode("utf-8")
        exitcode = completed.returncode
        m = evbound_re.search(stdout)
        evbound = m.group(1) if m else None
        m = tailbound_re.search(stdout)
        tailbound = m.group(1) if m else None
        result = BenchmarkResult(
            tool,
            path,
            flags,
            time=elapsed,
            stdout=stdout,
            stderr=stderr,
            exitcode=exitcode,
            timeout=timeout,
            evbound=evbound,
            tailbound=tailbound,
        )
        if exitcode != 0:
            result.exitcode = exitcode
            result.error = "crashed"
            if "Solver failed" in stderr:
                if "infeasible" in stderr:
                    result.error = "infeasible"
                if "unknown reason" in stderr:
                    result.error = "solver_error"
                if "timeout" in stderr:
                    result.error = "timeout"
            if "panicked" in stderr:
                result.error = "panic"
            print(
                f"Tool {tool} {red}FAILED ({result.error}){reset} with exit code {exitcode} in {elapsed:.3f}s."
            )
        else:
            m = total_time_re.search(stdout)
            if m:
                total_time = float(m.group(1))
            else:
                print(
                    f"Tool {tool} {red}did not output its total inference time{reset}. Using the total running time instead..."
                )
                result.time = elapsed
            print(
                f"Tool {tool} {green}successfully{reset} inferred {path} in {total_time:.4f}s"
            )
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8")
        stderr = (e.stderr or b"").decode("utf-8")
        print(f"Timemout of {timeout}s {red}expired{reset}.")
        result = BenchmarkResult(
            tool,
            path,
            flags,
            time=timeout,
            stdout=stdout,
            stderr=stderr,
            error="timeout",
        )
    return result


def bench_tool(tool, command, path: Path, timeout, flags=[]):
    flags = list(flags)
    if not path.is_file():
        return None
    file_contents = path.read_text()
    m = flags_re.search(file_contents)
    if m:
        flags += m.group(1).split()
    for m in tool_flags_re.finditer(file_contents):
        if tool.startswith(m.group(1).strip()):
            extra_flags = m.group(2).strip().split()
            print(f"Setting flags for {tool}: {extra_flags}")  # TODO: remove
            flags = extra_flags
    path_noext = path.with_suffix("")
    path_with_tool = Path(f"{path_noext}_{tool}.sgcl")
    for ext in [".out", ".err"]:
        if path_with_tool.with_suffix(ext).is_file():
            path_with_tool.with_suffix(ext).unlink()
    best_result = None
    for run in range(num_runs):
        result = run_tool(tool, command, path, flags, timeout)
        if best_result is None or (result.time < best_result.time and not result.error):
            best_result = result
    with open(path_with_tool.with_suffix(".out"), "w") as f:
        f.write(best_result.stdout)
    if best_result.stderr:
        with open(path_with_tool.with_suffix(".err"), "w") as f:
            f.write(best_result.stderr)
    print(
        f"Best time of {num_runs} runs of {tool} on {path} with flags {flags} was: {best_result.time:.4f}"
    )
    print()
    return best_result


def bench(name, timeout):
    path = Path(f"{name}.sgcl")
    if "# SKIP" in path.read_text():
        print(f"{red}Skipping {name}...{reset}")
        return {}

    result = {}
    for tool in tools:
        result[tool.name] = bench_tool(tool.name, tool.path, path, timeout, tool.flags)
    return result


def jsonserialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BenchmarkResult):
        return obj.__dict__
    return obj


if __name__ == "__main__":
    start = time.time()
    cargo_command = "cargo build --bin residual --bin geobound"  # TODO: Change to release
    print(f"Running `{cargo_command}`...")
    with subprocess.Popen(
        cargo_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        for line in process.stdout:
            print(line.decode("utf8"))
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    all_results = {}
    for benchmark_dir in benchmark_dirs:
        print(f"Benchmark suite {benchmark_dir}")
        print("===============")
        for benchmark in sorted(Path(benchmark_dir).iterdir()):
            if not benchmark.is_file() or benchmark.suffix != ".sgcl":
                continue
            benchmark = benchmark.with_suffix("")
            print(f"Benchmarking {benchmark}")
            print("------------")
            results = bench(benchmark, timeout)
            all_results[str(benchmark)] = results
            for tool, res in results.items():
                if res.error:
                    print(f"  {tool}: {red}{res.error}{reset}")
                else:
                    print(f"  {tool}: {green}{res.time:.5f}s{reset}")
                    print(f"  {tool}: EV bound: {res.evbound}")
                    print(f"  {tool}: tail bound: {res.tailbound}")
            print()
            print()
    end = time.time()
    elapsed = end - start
    print(f"{green}Benchmarking finished successfully in {elapsed:.1f}s.{reset}")

    successes = Counter()
    total = len(all_results)
    print(f"Summary of {len(all_results)} benchmarks:")
    print("===================")
    for benchmark, result in all_results.items():
        print(f"{benchmark}.sgcl:")
        for tool, res in result.items():
            if res.error:
                print(f"  {tool}: {red}{res.error}{reset}")
            else:
                print(f"  {tool}: {green}{res.time:.5f}s{reset}")
                print(f"  {tool}: EV bound: {res.evbound}")
                print(f"  {tool}: tail bound: {res.tailbound}")
                successes[tool] += 1
    print()
    for tool, successes in successes.items():
        print(
            f"{tool}: {green}{successes}{reset} / {total} = {round((successes / total) * 100)}% succeeded"
        )
    with open("bench-results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=jsonserialize)
    print(f"Results written to {own_path}/bench-results.json")
    print(f"Total time: {elapsed:.1f}s")
