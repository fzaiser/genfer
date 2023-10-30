#!/usr/bin/env python3

import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

green = "\033[92m"
red = "\033[91m"
reset = "\033[0m"

benchmarks = [
    "alarm",
    "clickGraph",
    "clinicalTrial",
    "clinicalTrial2",
    "digitRecognition",
    "evidence1",
    "evidence2",
    "grass",
    "murderMystery",
    "noisyOr",
    "twocoins",
]

ram_limit = 12 * 1024 * 1024 * 1024
stack_size = 64 * 1024 * 1024
timeout = 3600
num_runs = 5

inference_time_re = re.compile("Total inference time: ([0-9.]*)s")
flags_re = re.compile("flags: (.*)")


def set_limits():
    resource.setrlimit(resource.RLIMIT_AS, (ram_limit, resource.RLIM_INFINITY))
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack_size, hard))


def run_tool(tool, tool_command, path, expected, flags):
    if not isinstance(tool_command, list):
        tool_command = [tool_command]
    try:
        command = tool_command + flags + [path]
        print(f"Running {command}...")
        start = time.perf_counter()
        completed = subprocess.run(
            command, timeout=timeout, capture_output=True, preexec_fn=set_limits
        )
        elapsed = time.perf_counter() - start
        output = (completed.stdout or b"").decode("utf-8")
        stderr = (completed.stderr or b"").decode("utf-8")
        exitcode = completed.returncode
        if exitcode != 0:
            print(
                f"Tool {tool} {red}FAILED{reset} (exit code {exitcode}) in {elapsed:.3f}s.\nStdout:\n{output}\nStderr:\n{stderr}"
            )
            return "crashed"
        else:
            m = inference_time_re.search(output)
            if m:
                inference_time = float(m.group(1))
            else:
                print(f"Tool {tool} {red}did not output its total inference time{reset}. Using the total running time instead...")
                inference_time = elapsed
            if any(e in output for e in expected):
                print(
                    f"Tool {tool} {green}correctly{reset} inferred {path} in {inference_time:.4f}s"
                )
                return inference_time
            else:
                print(
                    f"Tool {tool} inferred {red}INCORRECT{reset} output on {path} in {inference_time:.4f}s. Output:\n{output}"
                )
                return "incorrect"
    except subprocess.TimeoutExpired:
        print(f"Timemout of {timeout}s {red}expired{reset}.")
        return "timeout"


def bench_tool(tool, command, path: Path, expected):
    if not path.is_file():
        return "n/a"
    m = flags_re.search(path.read_text())
    if m:
        flags = m.group(1).split()
    else:
        flags = []
    best_result = None
    for run in range(num_runs):
        result = run_tool(tool, command, path, expected, flags)
        if not isinstance(result, float):
            best_result = result
            break
        if best_result is None or result < best_result:
            best_result = result
    best = f"{best_result:.4f}s" if isinstance(best_result, float) else best_result
    print(f"Best of {num_runs} runs of {tool} on {path} with flags {flags} was: {best}")
    print()
    return best_result


def bench(name):
    expected = Path(f"benchmarks/{name}.expected").read_text().strip().splitlines()
    expected = list(e for e in expected if e != "")
    assert len(expected) > 0, f"No expected string for {name} found."
    genfer = bench_tool("Genfer", genfer_path, Path(f"benchmarks/{name}.sgcl"), expected)
    dice = bench_tool("Dice", dice_path, Path(f"benchmarks/{name}.dice"), expected)
    genfer_rational = bench_tool(
        "Genfer (rational)",
        [genfer_path, "--rational"],
        Path(f"benchmarks/{name}.rational.sgcl")
        if Path(f"benchmarks/{name}.rational.sgcl").is_file()
        else Path(f"benchmarks/{name}.sgcl"),
        expected,
    )
    dice_rational = bench_tool("Dice (rational)", [dice_path, "-wmc-type", "1"], Path(f"benchmarks/{name}.dice"), expected)
    prodigy = bench_tool("Prodigy", ["bash", "prodigy.sh"], Path(f"benchmarks/{name}.pgcl"), expected)
    psi = bench_tool("PSI", psi_path, Path(f"benchmarks/{name}.psi"), expected)
    return {
        "genfer": genfer,
        "dice": dice,
        "genfer-rational": genfer_rational,
        "dice-rational": dice_rational,
        "prodigy": prodigy,
        "psi": psi,
    }


if __name__ == "__main__":
    start = time.time()
    genfer_path = os.environ.get("GENFER", "../genfer/target/release/genfer")
    prodigy_path = "prodigy"
    dice_path = os.environ.get("DICE", "dice")
    psi_path = os.environ.get("PSI", "psi")
    own_path = Path(sys.argv[0]).parent
    os.chdir(own_path)
    all_results = {}
    for benchmark in benchmarks:
        print(f"Benchmarking {benchmark}")
        print("============")
        results = bench(benchmark)
        all_results[benchmark] = results
        print()
        print()
    with open("bench-results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    end = time.time()
    elapsed = end - start
    print(f"{green}Benchmarking finished successfully in {elapsed:.1f}s.{reset}")
