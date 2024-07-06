#!/usr/bin/env python3

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

timeout = 10
num_runs = 1  # TODO: increase

total_time_re = re.compile("Total time: ([0-9.]*)s")
flags_re = re.compile("flags: (.*)")


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
    ):
        self.tool = tool
        self.path = path
        self.flags = flags
        self.stdout = stdout
        self.stderr = stderr
        self.time = time
        self.error = error
        self.exitcode = exitcode


def run_tool(tool, tool_command, path, flags):
    if not isinstance(tool_command, list):
        tool_command = [tool_command]
    try:
        command = tool_command + flags + [path]
        print(f"Running {command}...")
        start = time.perf_counter()
        completed = subprocess.run(command, timeout=timeout, capture_output=True)
        elapsed = time.perf_counter() - start
        stdout = (completed.stdout or b"").decode("utf-8")
        stderr = (completed.stderr or b"").decode("utf-8")
        exitcode = completed.returncode
        result = BenchmarkResult(
            tool,
            path,
            flags,
            time=elapsed,
            stdout=stdout,
            stderr=stderr,
            exitcode=exitcode,
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
                f"Tool {tool} {red}FAILED ({result.error}){reset} in {elapsed:.3f}s.\nStdout:\n{stdout}\nStderr:\n{stderr}"
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


def bench_tool(tool, command, path: Path, flags=[]):
    if not path.is_file():
        return None
    m = flags_re.search(path.read_text())
    if m:
        flags = m.group(1).split()
    for ext in [".out", ".err"]:
        if path.with_suffix(ext).is_file():
            path.with_suffix(ext).unlink()
    best_result = None
    for run in range(num_runs):
        result = run_tool(tool, command, path, flags)
        if best_result is None or (result.time < best_result.time and not result.error):
            best_result = result
    with open(path.with_suffix(".out"), "w") as f:
        f.write(best_result.stdout)
    if best_result.stderr:
        with open(path.with_suffix(".err"), "w") as f:
            f.write(best_result.stderr)
    print(
        f"Best time of {num_runs} runs of {tool} on {path} with flags {flags} was: {best_result.time:.4f}"
    )
    print()
    return best_result


def bench(name):
    path = Path(f"{name}.sgcl")
    if "# SKIP" in path.read_text():
        print(f"{red}Skipping {name}...{reset}")
        return {}

    residual = bench_tool("residual", residual_path, path)
    geo_bound = bench_tool(
        "geo_bound",
        geo_bound_path,
        path,
        flags="-u 0 --solver ipopt --keep-while".split(),
    )
    return {
        "residual": residual,
        "geo_bound": geo_bound,
    }


def env(name, default):
    if name in os.environ:
        return os.environ[name]
    else:
        print(
            f"{red}Environment variable `{name}` is not set!{reset} Defaulting to `{default}`"
        )
        return default


def jsonserialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BenchmarkResult):
        return obj.__dict__
    return obj


if __name__ == "__main__":
    start = time.time()
    cargo_command = "cargo build --bin residual --bin bound"  # TODO: Change to release
    print(f"Running `{cargo_command}`...")
    with subprocess.Popen(
        cargo_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        for line in process.stdout:
            print(line.decode("utf8"))
    residual_path = env(
        "RESIDUAL", "../../target/debug/residual"
    )  # TODO: Change to release
    geo_bound_path = env(
        "GEO_BOUND", "../../target/debug/bound"
    )  # TODO: Change to release
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
            results = bench(benchmark)
            all_results[str(benchmark)] = results
            for tool, res in results.items():
                if res.error:
                    print(f"  {tool}: {red}{res.error}{reset}")
                else:
                    print(f"  {tool}: {green}{res.time:.5f}s{reset}")
            print()
            print()
    end = time.time()
    elapsed = end - start
    print(f"{green}Benchmarking finished successfully in {elapsed:.1f}s.{reset}")

    total = 0
    successes = 0
    print(f"Summary of {len(all_results)} benchmarks:")
    print("===================")
    for benchmark, result in all_results.items():
        print(f"{benchmark}.sgcl:")
        for tool, res in result.items():
            total += 1
            if res.error:
                print(f"  {tool}: {red}{res.error}{reset}")
            else:
                print(f"  {tool}: {green}{res.time:.5f}s{reset}")
                successes += 1
    print()
    print(
        f"{green}{successes} benchmark runs succeeded{reset} and {red}{total - successes} failed{reset}."
    )
    with open("bench-results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=jsonserialize)
