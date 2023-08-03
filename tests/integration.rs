use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;

use expect_test::expect_file;
use walkdir::WalkDir;

fn find_flags_in_first_line(input_path: &Path) -> Vec<String> {
    let first_line = BufReader::new(File::open(input_path).unwrap())
        .lines()
        .next()
        .unwrap()
        .unwrap();
    if let Some(pos) = first_line.find("flags: ") {
        first_line[pos + 7..]
            .trim()
            .split_ascii_whitespace()
            .map(|f| f.trim().to_owned())
            .collect()
    } else {
        vec![]
    }
}

fn check_output(input_file: impl AsRef<Path>, expected_output_file: impl AsRef<Path>) {
    let input_path = input_file.as_ref();
    let binary_path = env!("CARGO_BIN_EXE_genfer");
    let mut command = Command::new(binary_path);
    command
        .env("RUST_BACKTRACE", "1")
        .arg(input_path)
        .arg("--no-timing");
    find_flags_in_first_line(input_path).iter().for_each(|f| {
        command.arg(f);
    });
    let output = command.output().unwrap();
    if !output.status.success() {
        panic!(
            "The command {:?} failed with the following output: {}",
            command,
            String::from_utf8(output.stderr).unwrap()
        )
    }
    let output = String::from_utf8(output.stdout).unwrap();
    let expected_output = expect_file![expected_output_file.as_ref()];
    expected_output.assert_eq(&output);
}

fn check_dir(dir: &str) {
    let test_expect_dir = format!("{}/test/expect/{dir}", env!("CARGO_MANIFEST_DIR"));
    for entry in WalkDir::new(test_expect_dir) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        // Check if file name has the extension ".sgcl" and if so,
        // check that it has a corresponding ".expect" file
        if path.extension() == Some("sgcl".as_ref()) {
            println!("Testing {} ...", path.display());
            let expect_path = path.with_extension("expect");
            check_output(path, &expect_path);
        }
    }
}

#[test]
fn expect_tests_sample() {
    check_dir("sample");
}

#[test]
fn expect_tests_observe() {
    check_dir("observe");
}

#[test]
fn expect_tests_if() {
    check_dir("if");
}

#[test]
fn expect_tests_assign() {
    check_dir("assign");
}

#[test]
fn expect_tests_normalize() {
    check_dir("normalize");
}

#[test]
fn expect_tests_examples() {
    check_dir("examples");
}

#[test]
fn expect_tests_former_bugs() {
    check_dir("former_bugs");
}

#[test]
fn expect_tests_real_world() {
    check_dir("real_world");
}

#[test]
fn expect_tests_slow() {
    if std::env::var("RUN_SLOW_TESTS").is_err() {
        return;
    }
    check_dir("slow");
}
