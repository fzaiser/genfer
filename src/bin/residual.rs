#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use genfer::bounds::residual::ResidualSemantics;
use genfer::interval::Interval;
use genfer::number::{Rational, F64};
use genfer::parser;
use genfer::ppl::{Program, Var};
use genfer::semantics::support::VarSupport;
use genfer::semantics::Transformer;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use ndarray::Axis;
use num_traits::{One, Zero};

#[allow(clippy::struct_excessive_bools)]
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// The file containing the probabilistic program
    file_name: PathBuf,
    /// Print the parsed probabilistic program
    #[arg(long)]
    print_program: bool,
    /// Disable timing of the execution
    #[arg(long)]
    no_timing: bool,
    /// Disable normalization of the distribution
    #[arg(long)]
    no_normalize: bool,
    #[arg(short = 'u', long, default_value = "8")]
    /// The default number of loop unrollings
    unroll: usize,
    /// The limit for the probability masses to be computed
    #[arg(short, long)]
    pub limit: Option<usize>,
    /// Whether to print the generated constraints
    #[arg(short, long)]
    verbose: bool,
}

pub fn main() -> std::io::Result<()> {
    let args = CliArgs::parse();
    let contents = std::fs::read_to_string(&args.file_name)?;
    let program = parser::parse_program(&contents);
    if args.print_program {
        println!("Parsed program:\n{program}\n");
    }
    run_program(&program, &args)?;
    Ok(())
}

fn run_program(program: &Program, args: &CliArgs) -> std::io::Result<()> {
    let start = Instant::now();
    let mut sem = ResidualSemantics::default()
        .with_verbose(args.verbose)
        .with_unroll(args.unroll);
    let mut result = sem.semantics(program);
    if args.verbose {
        match &result.var_supports {
            VarSupport::Empty(_) => println!("Support: empty"),
            VarSupport::Prod(supports) => {
                for (v, support) in supports.iter().enumerate() {
                    println!("Support of {}: {support}", Var(v));
                }
            }
        }
        println!();
        println!("Bound result:");
        println!("{result}");
    }
    let residual = result.residual();
    let support = result.var_supports[program.result].clone();

    for v in 0..result.var_supports.num_vars() {
        if Var(v) != program.result {
            result = result.marginalize(Var(v));
        }
    }
    if args.verbose {
        println!("\nMarginalized bound:");
        let ax = Axis(program.result.id());
        for i in 0..result.lower.masses.len_of(ax) {
            let lo = result
                .lower
                .masses
                .index_axis(ax, i)
                .first()
                .unwrap()
                .clone();
            let hi = lo.clone() + residual.clone();
            println!("{i}: [{}, {}]", lo.round_to_f64(), hi.round_to_f64());
        }
    }

    let (norm_lo, norm_hi) = if !args.no_normalize && program.uses_observe() {
        let total_lo = result.lower.total_mass().clone();
        let total_hi = Rational::one() - result.reject.clone();
        if args.verbose {
            println!(
                "\nNormalizing constant: Z {}",
                in_iv(&Interval::exact(total_lo.clone(), total_hi.clone()))
            );
            println!("Everything from here on is normalized.");
        }
        (total_lo, total_hi)
    } else {
        (Rational::one(), Rational::one())
    };

    println!("\nProbability masses:");
    let limit = if let Some(range) = result.var_supports[program.result].finite_nonempty_range() {
        *range.end() as usize + 1
    } else {
        args.limit.unwrap_or(20)
    };
    let lower_probs = result.lower.probs(program.result);
    for i in 0..limit {
        let lo = lower_probs.get(i).unwrap_or(&Rational::zero()).clone();
        let hi = if support.contains(i as u32) {
            lo.clone() + residual.clone()
        } else {
            assert!(lo.is_zero());
            Rational::zero()
        };
        let prob = Interval::exact(lo / norm_hi.clone(), hi / norm_lo.clone());
        println!("p({i}) {}", in_iv(&prob));
    }
    println!(
        "p(n) {} for all n >= {limit}",
        in_iv(&Interval::exact(Rational::zero(), residual.clone()))
    );

    println!("\nMoments:");
    let lower_moments = result.lower.moments(program.result, 5);
    let range = support.to_interval().unwrap_or(Interval::zero());
    for i in 0..5 {
        let added = residual.clone() * range.hi.clone().pow(i as i32);
        let lo = lower_moments[i].clone();
        let hi = lo.clone() + added.clone();
        let moment = Interval::exact(lo / norm_hi.clone(), hi / norm_lo.clone());
        println!("{i}-th (raw) moment {}", in_iv(&moment));
    }
    println!("Total time: {:.5}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn in_iv(iv: &Interval<Rational>) -> String {
    if iv.lo == iv.hi {
        format!("= {}", iv.lo.round_to_f64())
    } else {
        format!(
            "∈ [{}, {}]",
            F64::from(iv.lo.round_down()),
            F64::from(iv.hi.round_up())
        )
    }
}
