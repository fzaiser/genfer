#!/bin/bash

cd ..
cargo build
cd benchmarks/

../target/debug/residual herman.sgcl --limit 200 -u 30 > outputs/herman_residual.txt
../target/debug/geobound herman.sgcl --limit 200 --optimizer ipopt --objective tail > outputs/herman_bound_tail.txt
../target/debug/geobound herman.sgcl --limit 200 -u 30 --optimizer ipopt --objective ev > outputs/herman_bound_moments.txt
../target/debug/geobound herman.sgcl --limit 200 -u 30 --optimizer ipopt --objective total --keep-while > outputs/herman_bound_probs.txt

../target/debug/residual coupon-collector.sgcl --limit 300 -u 50 > outputs/coupon_collector_residual.txt
../target/debug/geobound coupon-collector.sgcl --limit 300 -u 20 --optimizer ipopt --objective tail > outputs/coupon_collector_bound_tail.txt
../target/debug/geobound coupon-collector.sgcl --limit 300 -u 50 --optimizer ipopt --objective ev > outputs/coupon_collector_bound_moments.txt
../target/debug/geobound coupon-collector.sgcl --limit 300 -u 50 --optimizer ipopt --objective total > outputs/coupon_collector_bound_probs.txt

../target/debug/residual geo.sgcl --limit 200 -u 50 > outputs/geo_residual.txt
../target/debug/geobound geo.sgcl --limit 200 --solver ipopt --optimizer adam-barrier --objective tail > outputs/geo_bound_tail.txt
../target/debug/geobound geo.sgcl  --limit 200 -u 50 --solver ipopt --optimizer ipopt --optimizer linear --objective ev > outputs/geo_bound_moments.txt
../target/debug/geobound geo.sgcl  --limit 200 -u 50 --solver adam-barrier --optimizer adam-barrier --keep-while --objective total > outputs/geo_bound_probs.txt

../target/debug/residual asym_rw.sgcl --limit 200 -u 40 > outputs/asym_rw_residual.txt
../target/debug/geobound asym_rw.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier --objective tail > outputs/asym_rw_bound_tail.txt
../target/debug/geobound asym_rw.sgcl --limit 200 -u 50 -d 3 --solver adam-barrier --optimizer adam-barrier --optimizer linear --objective ev > outputs/asym_rw_bound_moments.txt
../target/debug/geobound asym_rw.sgcl --limit 200 -u 40 -d 3 --solver adam-barrier --optimizer adam-barrier --optimizer linear --objective total > outputs/asym_rw_bound_probs.txt

../target/debug/residual die_paradox.sgcl --limit 200 -u 40 > outputs/die_paradox_residual.txt
../target/debug/geobound die_paradox.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier --objective tail > outputs/die_paradox_bound_tail.txt
../target/debug/geobound die_paradox.sgcl --limit 200 -u 40 --solver ipopt --optimizer ipopt --optimizer linear --objective ev > outputs/die_paradox_bound_moments.txt
../target/debug/geobound die_paradox.sgcl --limit 200 -u 40 --solver adam-barrier --optimizer adam-barrier --optimizer linear --keep-while --objective total > outputs/die_paradox_bound_probs.txt
