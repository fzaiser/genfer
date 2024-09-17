#!/bin/bash

cd ..
cargo build
cd benchmarks/

../target/debug/residual geo.sgcl --limit 200 -u 50 > outputs/geo_residual.txt
../target/debug/geobound geo.sgcl --limit 200 --solver ipopt --optimizer adam-barrier --objective tail > outputs/geo_bound_tail.txt
../target/debug/geobound geo.sgcl  --limit 200 -u 50 --solver ipopt --optimizer ipopt --objective ev > outputs/geo_bound_moments.txt
../target/debug/geobound geo.sgcl  --limit 200 -u 50 --solver adam-barrier --optimizer adam-barrier --keep-while > outputs/geo_bound_probs.txt

../target/debug/residual asym_rw.sgcl --limit 200 -u 40 > outputs/asym_rw_residual.txt
../target/debug/geobound asym_rw.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier --objective tail > outputs/asym_rw_bound_tail.txt
../target/debug/geobound asym_rw.sgcl --limit 200 -u 50 -d 3 --solver adam-barrier --optimizer adam-barrier --objective ev > outputs/asym_rw_bound_moments.txt
../target/debug/geobound asym_rw.sgcl --limit 200 -u 40 -d 3 --solver adam-barrier --optimizer adam-barrier > outputs/asym_rw_bound_probs.txt

../target/debug/residual die_paradox.sgcl --limit 200 -u 40 > outputs/die_paradox_residual.txt
../target/debug/geobound die_paradox.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier --objective tail > outputs/die_paradox_bound_tail.txt
../target/debug/geobound die_paradox.sgcl --limit 200 -u 40 --solver ipopt --optimizer ipopt --objective ev > outputs/die_paradox_bound_moments.txt
../target/debug/geobound die_paradox.sgcl --limit 200 -u 40 --solver adam-barrier --optimizer adam-barrier --keep-while > outputs/die_paradox_bound_probs.txt
