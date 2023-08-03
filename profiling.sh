#!/usr/bin/env bash

echo "USAGE: ./profiling.sh <binary> <args>"

set -euo pipefail
IFS=$'\n\t'

cargo build --profile release-dev
echo "RUNNING perf ON $@"
perf record --call-graph=dwarf ./target/release-dev/$@
echo "DONE. CONVERTING TO TEXT FILE profile.perf..."
perf script -F +pid > profile.perf
echo "DONE. You can view profile.perf on https://profiler.firefox.com/"
