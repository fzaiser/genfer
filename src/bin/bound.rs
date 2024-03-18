#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use genfer::bounds::ctx::{BoundCtx, SolverError};
use genfer::number::F64;
use genfer::parser;
use genfer::ppl::{Program, Var};
use genfer::semantics::support::VarSupport;
use genfer::semantics::Transformer;

use clap::Parser;
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
    /// Timeout for the SMT solver in ms
    #[arg(long, default_value = "30000")]
    timeout: u64,
    /// Whether to optimize the bound once one is found
    #[arg(long)]
    no_optimize: bool,
    /// Optionally output an SMT-LIB file at this path
    #[arg(long)]
    smt: Option<PathBuf>,
    /// Optionally output a QEPCAD file (to feed to qepcad via stdin) at this path
    #[arg(long)]
    qepcad: Option<PathBuf>,
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
    let mut ctx = BoundCtx::new();
    let result = ctx.semantics(program);
    match &result.var_supports {
        VarSupport::Empty(_) => println!("Support: empty"),
        VarSupport::Prod(supports) => {
            for (v, support) in supports.iter().enumerate() {
                println!("Support of {}: {support}", Var(v));
            }
        }
    }
    println!("Bound result:");
    println!("{result}");
    println!("Constraints:");
    for constraint in ctx.constraints() {
        println!("  {constraint}");
    }
    // println!("Python:");
    // println!("{}", ctx.output_python_z3());
    if let Some(path) = &args.smt {
        println!("Writing SMT file to {path:?}...");
        let mut out = std::fs::File::create(path)?;
        ctx.output_smt(&mut out)?;
    }
    if let Some(path) = &args.qepcad {
        println!("Writing QEPCAD commands to {path:?}...");
        let mut out = std::fs::File::create(path)?;
        ctx.output_qepcad(&mut out)?;
    }
    let time_constraint_gen = start.elapsed();
    println!("Constraint generation time: {time_constraint_gen:?}");
    println!("Solving constraints...");
    let start_smt = Instant::now();
    let solver_result = ctx.solve_z3(
        &result.bound,
        Duration::from_millis(args.timeout),
        !args.no_optimize,
    );
    let solver_time = start_smt.elapsed();
    println!("Solver time: {solver_time:?}");
    match solver_result {
        Ok(z3_bound) => {
            println!("\nFinal bound:\n");
            println!("{z3_bound}");

            println!("\nProbability masses:");
            let mut inputs = vec![F64::one(); result.var_supports.num_vars()];
            inputs[program.result.id()] = F64::zero();
            let degree_p1 =
                if let Some(range) = result.var_supports[program.result].finite_nonempty_range() {
                    *range.end() as usize + 1
                } else {
                    100
                };
            let expansion = z3_bound.evaluate_var(&inputs, program.result, degree_p1);
            let mut index = vec![0; result.var_supports.num_vars()];
            for i in 0..degree_p1 {
                index[program.result.id()] = i;
                let prob = expansion.coefficient(&index);
                println!("p({i}) <= {prob}");
            }

            println!("\nMoments:");
            let inputs = vec![F64::one(); result.var_supports.num_vars()];
            let expansion = z3_bound.evaluate_var(&inputs, program.result, 5);
            let mut index = vec![0; result.var_supports.num_vars()];
            for i in 0..5 {
                index[program.result.id()] = i;
                let factorial_moment = expansion.coefficient(&index);
                println!("{i}-th factorial moment <= {factorial_moment}");
            }
        }
        Err(e) => match e {
            SolverError::Timeout => {
                println!("SMT solver timeout");
            }
            SolverError::Infeasible => {
                println!("SMT proved that there is no bound of the required form");
            }
        },
    }
    println!("Total time: {:?}", start.elapsed());
    Ok(())
}
