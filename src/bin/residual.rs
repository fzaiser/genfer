#![warn(clippy::pedantic)]
#![expect(clippy::needless_range_loop)]
#![expect(clippy::cast_possible_truncation)]

use std::path::PathBuf;
use std::time::Instant;
use tool::interval::Interval;
use tool::numbers::{Rational, F64};
use tool::parser;
use tool::ppl::{Program, Var};
use tool::semantics::residual::ResidualSemantics;
use tool::semantics::support::VarSupport;
use tool::semantics::Transformer;

use clap::Parser;
use ndarray::Axis;
use num_traits::{One, Zero};

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
    run_program(&program, &args);
    Ok(())
}

fn run_program(program: &Program, args: &CliArgs) {
    let start = Instant::now();
    let mut sem = ResidualSemantics::default()
        .with_verbose(args.verbose)
        .with_unroll(args.unroll);
    let result = sem.semantics(program);
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

    let result = result.marginal(program.result);

    println!("\nUnnormalized bounds:");
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
        println!("p'({i}) {}", in_iv(&Interval::exact(lo, hi)));
    }
    println!(
        "p'(n) {} for all n >= {}",
        in_iv(&Interval::exact(Rational::zero(), residual.clone())),
        result.lower.masses.len_of(ax)
    );

    let norm = if program.uses_observe() {
        let total_lo = result.lower.total_mass().clone();
        let total_hi = Rational::one() - result.reject.clone();
        let total = Interval::exact(total_lo.clone(), total_hi.clone());
        if args.verbose {
            println!("\nNormalizing constant: Z {}", in_iv(&total));
            println!("Everything from here on is normalized.");
        }
        total
    } else {
        Interval::one()
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
        let hi = if support.contains(i as u64) {
            lo.clone() + residual.clone()
        } else {
            assert!(lo.is_zero());
            Rational::zero()
        };
        let prob = Interval::exact(lo, hi) / norm.clone();
        println!("p({i}) {}", in_iv(&prob));
    }
    let residual_norm = Interval::exact(Rational::zero(), residual.clone()) / norm.clone();
    println!("p(n) {} for all n >= {limit}", in_iv(&residual_norm));

    println!("\nMoments:");
    let lower_moments = result.lower.moments(program.result, 5);
    let range = support.to_interval().unwrap_or(Interval::zero());
    for i in 0..5 {
        let added = residual.clone() * range.hi.clone().pow(i32::try_from(i).unwrap());
        let lo = lower_moments[i].clone();
        let hi = lo.clone() + added.clone();
        let moment = Interval::exact(lo, hi) / norm.clone();
        println!("{i}-th (raw) moment {}", in_iv(&moment));
    }
    println!("Total time: {:.5}s", start.elapsed().as_secs_f64());
}

fn in_iv(iv: &Interval<Rational>) -> String {
    if iv.lo == iv.hi {
        format!("= {}", iv.lo.to_f64())
    } else {
        format!(
            "∈ [{}, {}]",
            F64::from(iv.lo.to_f64_down()),
            F64::from(iv.hi.to_f64_up())
        )
    }
}
