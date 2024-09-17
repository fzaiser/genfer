use std::path::PathBuf;

use clap::Parser;
use tool::parser::parse_program;
use tool::semantics::support::{SupportTransformer, VarSupport};
use tool::semantics::Transformer;
use tool::support::SupportSet;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// The file containing the probabilistic program
    file_name: PathBuf,
}

pub fn main() {
    let args = CliArgs::parse();
    let path = args.file_name;
    let contents = std::fs::read_to_string(path).unwrap();
    let program = parse_program(&contents);
    let support = SupportTransformer::default().semantics(&program);
    println!(
        "{} variables, {} statements (including nesting)",
        support.num_vars(),
        program.size()
    );
    println!("Support: {support}");
    let support_size = match support {
        VarSupport::Empty(_) => Some(0),
        VarSupport::Prod(supports) => {
            supports
                .iter()
                .try_fold(1, |acc, support| match (acc, support) {
                    (
                        acc,
                        SupportSet::Range {
                            start,
                            end: Some(end),
                        },
                    ) => Some(acc * u128::from(end - start + 1)),
                    _ => None,
                })
        }
    };
    println!(
        "Support size: {}",
        if let Some(size) = support_size {
            format!("{size}")
        } else {
            "infinite".to_owned()
        }
    );
    println!("Contains observations: {}", program.uses_observe());
}
