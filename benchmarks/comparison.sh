#!/bin/bash

cd ..
cargo build --release --bins
cd benchmarks/


echo "Computing bounds for geo.sgcl ..."
../target/release/residual geo.sgcl --limit 200 -u 50 > outputs/geo_residual.txt
../target/release/geobound geo.sgcl --limit 200 -u 1 --objective tail > outputs/geo_bound_tail.txt
../target/release/geobound geo.sgcl --limit 200 -u 50 --objective ev > outputs/geo_bound_moments.txt
../target/release/geobound geo.sgcl --limit 200 -u 50 --objective total > outputs/geo_bound_probs.txt

echo "Computing bounds for asym_rw.sgcl ..."
../target/release/residual asym_rw.sgcl --limit 300 -u 70 > outputs/asym_rw_residual.txt
../target/release/geobound asym_rw.sgcl --limit 300 -u 1 -d 2 --objective tail > outputs/asym_rw_bound_tail.txt
../target/release/geobound asym_rw.sgcl --limit 300 -u 70 -d 2 --objective ev > outputs/asym_rw_bound_moments.txt
../target/release/geobound asym_rw.sgcl --limit 300 -u 70 -d 2 --objective total > outputs/asym_rw_bound_probs.txt

echo "Computing bounds for die_paradox.sgcl ..."
../target/release/residual die_paradox.sgcl --limit 200 -u 40 > outputs/die_paradox_residual.txt
../target/release/geobound die_paradox.sgcl --limit 200 -u 1 -d 2 --objective tail > outputs/die_paradox_bound_tail.txt
../target/release/geobound die_paradox.sgcl --limit 200 -u 40 --objective ev > outputs/die_paradox_bound_moments.txt
../target/release/geobound die_paradox.sgcl --limit 200 -u 40 --keep-while --objective total > outputs/die_paradox_bound_probs.txt

echo "Computing bounds for coupon-collector.sgcl ..."
../target/release/residual coupon-collector.sgcl --limit 300 -u 80 > outputs/coupon_collector_residual.txt
../target/release/geobound coupon-collector.sgcl --limit 300 -u 1 --objective tail > outputs/coupon_collector_bound_tail.txt
../target/release/geobound coupon-collector.sgcl --limit 300 -u 80 --objective ev --optimizer ipopt --optimizer linear > outputs/coupon_collector_bound_moments.txt
../target/release/geobound coupon-collector.sgcl --limit 300 -u 80 --objective total --optimizer ipopt --optimizer linear > outputs/coupon_collector_bound_probs.txt

echo "Computing bounds for herman.sgcl ..."
../target/release/residual herman.sgcl --limit 200 -u 30 > outputs/herman_residual.txt
../target/release/geobound herman.sgcl --limit 200 -u 1 --objective tail > outputs/herman_bound_tail.txt
../target/release/geobound herman.sgcl --limit 200 -u 30 --optimizer ipopt --optimizer linear --objective ev > outputs/herman_bound_moments.txt
../target/release/geobound herman.sgcl --limit 200 -u 30 --optimizer ipopt --optimizer linear --objective total > outputs/herman_bound_probs.txt
