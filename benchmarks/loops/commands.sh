cargo run --bin residual -- geo.sgcl -u 50
cargo run --bin bound -- geo.sgcl --solver ipopt --optimizer adam-barrier --objective tail --limit 200
cargo run --bin bound -- geo.sgcl -u 50 --solver ipopt --optimizer ipopt --objective ev
cargo run --bin bound -- geo.sgcl -u 50 --solver adam-barrier --optimizer adam-barrier --limit 200 --keep-while

cargo run --bin residual -- asym_rw.sgcl -u 40
cargo run --bin bound -- asym_rw.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier --objective tail
cargo run --bin bound -- asym_rw.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier -u 50 --objective ev -d 3
cargo run --bin bound -- asym_rw.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier -u 40 -d 3

cargo run --bin residual -- die_paradox.sgcl -u 40
cargo run --bin bound -- die_paradox.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier --objective tail
cargo run --bin bound -- die_paradox.sgcl --limit 200 --solver ipopt --optimizer ipopt -u 40 --objective ev
cargo run --bin bound -- die_paradox.sgcl --limit 200 --solver adam-barrier --optimizer adam-barrier -u 40
