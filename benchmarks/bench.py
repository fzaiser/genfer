#!/usr/bin/env python3

from collections import Counter
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
reset = "\033[0m"

benchmark_dirs = [
    "polar",
    "prodigy",
    "psi",
    "ours",
]

timeout = 300  # seconds
num_runs = 3  # take the fastest of `num_runs` runs

total_time_re = re.compile(r"(?:Total|Elapsed) time: ([0-9.]*) *s")
flags_re = re.compile(r"flags: (.*)")
tool_flags_re = re.compile(r"flags ?\((.*)\): (.*)")
evbound_re = re.compile(r"(?:1-th \(raw\) moment|E\([A-Za-z0-9_]*\)) (.*)")
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
    def __init__(self, name, path, flags=[], file_ext=".sgcl"):
        self.name = name
        self.path = path
        self.flags = flags
        self.file_ext = file_ext


residual_path = "../target/release/residual"
geo_bound_path = "../target/release/geobound"
polar_path = os.environ.get("POLAR_PATH", "../../polar")
tools = [
    Tool("geobound-existence", geo_bound_path, ["-u", "0"]),
    Tool("geobound-ev", geo_bound_path, ["-u", "30", "--objective", "ev"]),
    Tool("geobound-tail", geo_bound_path, ["-u", "1", "--objective", "tail"]),
    Tool("polar", [fr"{polar_path}/.venv/bin/python", fr"{polar_path}/polar.py"], ["--after_loop"], file_ext=".prob"),
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
        command = tool_command + [path] + flags
        command_str = " ".join(str(x) for x in command)
        print(f"Running {command_str}...")
        start = time.perf_counter()
        env = os.environ.copy()
        env["RUST_BACKTRACE"] = "1"
        completed = subprocess.run(
            command, timeout=timeout, capture_output=True, env=env
        )
        measured_time = time.perf_counter() - start
        stdout = (completed.stdout or b"").decode("utf-8")
        stderr = (completed.stderr or b"").decode("utf-8")
        exitcode = completed.returncode
        m = total_time_re.search(stdout)
        if m:
            elapsed = float(m.group(1))
        else:
            print(
                f"Tool {tool} {red}did not output its total inference time{reset}. Using the total running time instead..."
            )
            elapsed = measured_time
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
        if exitcode == 0:
            print(
                f"Tool {tool} {green}successfully{reset} inferred {path} in {result.time:.4f} s"
            )
        else:
            result.exitcode = exitcode
            result.error = "failed"
            if "Solver failed" in stderr:
                if "infeasible" in stderr:
                    result.error = "possibly_infeasible"
                if "unknown reason" in stderr:
                    result.error = "solver_error"
            if "panicked" in stderr:
                result.error = "panic"
            print(
                f"Tool {tool} {red}FAILED ({result.error}){reset} with exit code {exitcode} in {measured_time:.3f} s."
            )
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode("utf-8")
        stderr = (e.stderr or b"").decode("utf-8")
        print(f"Timemout of {timeout} s {red}expired{reset}.")
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
    file_contents = path.read_text()
    m = flags_re.search(file_contents)
    if m:
        flags += m.group(1).split()
    for m in tool_flags_re.finditer(file_contents):
        if tool.startswith(m.group(1).strip()):
            extra_flags = m.group(2).strip().split()
            if extra_flags[0] == "-u":
                unroll_index = flags.index("-u")
                del flags[unroll_index : unroll_index + 2]
                flags = flags + extra_flags
            else:
                flags += extra_flags
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
            if result.error:
                print(f"Skipping remaining {num_runs - run - 1} runs...")
                break
    with open(path_with_tool.with_suffix(".out"), "w") as f:
        f.write(best_result.stdout)
    if best_result.stderr:
        with open(path_with_tool.with_suffix(".err"), "w") as f:
            f.write(best_result.stderr)
    print(
        f"Best time of {run + 1} runs of {tool} on {path} with flags {flags} was: {best_result.time:.4f} s"
    )
    print()
    return best_result


def bench(name, timeout):
    result = {}
    for tool in tools:
        path = Path(f"{name}{tool.file_ext}")
        if not path.is_file():
            print(f"{red}File {path} not found, skipping this benchmark for tool {tool.name} ...{reset}")
            continue
        if "# SKIP" in path.read_text():
            print(f"{red}Skipping benchmark {name} for tool {tool.name} ...{reset}")
            continue
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
    cargo_command = "cargo build --release --bins"
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
        print(f"{blue}Benchmark suite {benchmark_dir}{reset}")
        print("===============")
        for benchmark in sorted(Path(benchmark_dir).iterdir()):
            if not benchmark.is_file() or benchmark.suffix != ".sgcl":
                continue
            benchmark = benchmark.with_suffix("")
            print(f"{blue}Benchmarking {benchmark}{reset}")
            print("------------")
            results = bench(benchmark, timeout)
            all_results[str(benchmark)] = results
            for tool, res in results.items():
                if res.error:
                    print(f"  {tool}: {red}{res.error}{reset}")
                else:
                    print(f"  {tool}: {green}{res.time:.5f} s{reset}")
                    print(f"  {tool}: EV bound: {res.evbound}")
                    print(f"  {tool}: tail bound: {res.tailbound}")
            print()
            print()
    end = time.time()
    elapsed = end - start
    print(f"{green}Benchmarking finished successfully in {elapsed:.1f} s.{reset}")

    successes = Counter()
    skipped = Counter()
    total = len(all_results)
    print(f"{blue}Summary of {len(all_results)} benchmarks:{reset}")
    print("===================")
    for benchmark, result in all_results.items():
        print(f"{benchmark}:")
        for tool in tools:
            tool = tool.name
            if tool not in result:
                print(f"  {tool}: {red}SKIPPED{reset}")
                skipped[tool] += 1
                continue
            res = result[tool]
            if res.error:
                print(f"  {tool}: {red}{res.error}{reset}")
            else:
                print(f"  {tool}: {green}{res.time:.5f} s{reset}")
                print(f"  {tool}: EV bound: {res.evbound}")
                print(f"  {tool}: tail bound: {res.tailbound}")
                successes[tool] += 1
    print()
    for tool, successes in successes.items():
        total_run = total - skipped[tool]
        print(
            f"{tool}: {green}{successes}{reset} / {total_run} = {round((successes / total_run) * 100)}% succeeded ({skipped[tool]} skipped)"
        )
    with open("bench-results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=jsonserialize)
    print(f"Results written to {own_path}/bench-results.json")
    print(f"{blue}Total time: {elapsed:.1f} s{reset}")
