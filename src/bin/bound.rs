#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use genfer::bounds::ctx::BoundCtx;
use genfer::bounds::gradient_descent::{Adam, AdamBarrier, GradientDescent};
use genfer::bounds::optimizer::{LinearProgrammingOptimizer, Optimizer as _, Z3Optimizer};
use genfer::bounds::solver::{ConstraintProblem, Solver as _, SolverError, Z3Solver};
use genfer::multivariate_taylor::TaylorPoly;
use genfer::number::Rational;
use genfer::parser;
use genfer::ppl::{Program, Var};
use genfer::semantics::support::VarSupport;
use genfer::semantics::Transformer;

use clap::{Parser, ValueEnum};
use num_traits::One;

#[derive(Clone, ValueEnum)]
enum Solver {
    Z3,
    #[value(name = "gd")]
    GradientDescent,
    AdamBarrier,
}

#[derive(Clone, ValueEnum)]
enum Optimizer {
    Z3,
    #[value(name = "gd")]
    GradientDescent,
    Adam,
    AdamBarrier,
}

#[derive(Clone, ValueEnum)]
enum Objective {
    Total,
    #[value(name = "ev")]
    ExpectedValue,
    Tail,
}

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
    #[arg(short = 'd', long, default_value = "1")]
    /// The minimum degree of the loop invariant polynomial
    min_degree: usize,
    #[arg(short = 'u', long, default_value = "0")]
    /// The default number of loop unrollings
    unroll: usize,
    /// Timeout for the solver in ms
    #[arg(long, default_value = "10000")]
    timeout: u64,
    /// The solver to use
    #[arg(long, default_value = "z3")]
    solver: Solver,
    /// The optimizer to use
    #[arg(long)]
    optimizer: Option<Optimizer>,
    /// The optimization objective
    #[arg(long, default_value = "ev")]
    objective: Objective,
    #[arg(long)]
    evt: bool,
    /// Whether to optimize the linear parts of the bound at the end
    #[arg(long)]
    no_linear_optimize: bool,
    /// Don't transform while loops into do-while loops
    #[arg(long)]
    keep_while: bool,
    /// Optionally output an SMT-LIB file at this path
    #[arg(long)]
    smt: Option<PathBuf>,
    /// Optionally output a QEPCAD file (to feed to qepcad via stdin) at this path
    #[arg(long)]
    qepcad: Option<PathBuf>,
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
    let mut ctx = BoundCtx::new()
        .with_verbose(args.verbose)
        .with_min_degree(args.min_degree)
        .with_default_unroll(args.unroll)
        .with_evt(args.evt)
        .with_do_while_transform(!args.keep_while);
    let result = ctx.semantics(program);
    if args.verbose {
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
        println!("Polynomial constraints:");
        for constraint in ctx.constraints() {
            println!("  {}", constraint.to_poly());
        }
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
    let start_solver = Instant::now();
    let timeout = Duration::from_millis(args.timeout);
    let problem = ConstraintProblem {
        var_count: ctx.sym_var_count(),
        geom_vars: ctx.geom_vars().to_owned(),
        factor_vars: ctx.factor_vars().to_owned(),
        coefficient_vars: ctx.coefficient_vars().to_owned(),
        constraints: ctx.constraints().to_owned(),
    };
    let init_solution = match args.solver {
        Solver::Z3 => Z3Solver.solve(&problem, timeout),
        Solver::GradientDescent => GradientDescent::default().solve(&problem, timeout),
        Solver::AdamBarrier => AdamBarrier::default().solve(&problem, timeout),
    };
    let solver_time = start_solver.elapsed();
    println!("Solver time: {solver_time:?}");
    match init_solution {
        Ok(solution) => {
            let objective = match args.objective {
                Objective::Total => result.bound.total_mass(),
                Objective::ExpectedValue => result.bound.expected_value(program.result),
                Objective::Tail => result.bound.tail_objective(program.result),
            };
            println!("Optimizing solution...");
            let optimized_solution = if let Some(optimizer) = &args.optimizer {
                let start_optimizer = Instant::now();
                let optimized_solution = match optimizer {
                    Optimizer::Z3 => {
                        Z3Optimizer.optimize(&problem, &objective, solution.clone(), timeout)
                    }
                    Optimizer::GradientDescent => {
                        GradientDescent::default().optimize(&problem, &objective, solution, timeout)
                    }
                    Optimizer::Adam => {
                        Adam::default().optimize(&problem, &objective, solution, timeout)
                    }
                    Optimizer::AdamBarrier => {
                        AdamBarrier::default().optimize(&problem, &objective, solution, timeout)
                    }
                };
                let optimizer_time = start_optimizer.elapsed();
                println!("Optimizer time: {optimizer_time:?}");
                optimized_solution
            } else {
                solution
            };
            let optimized_solution = if args.no_linear_optimize {
                optimized_solution
            } else {
                LinearProgrammingOptimizer.optimize(
                    &problem,
                    &objective,
                    optimized_solution,
                    timeout,
                )
            };
            let bound = result.bound.resolve(&optimized_solution);
            println!("\nFinal bound:\n");
            println!("{bound}");

            println!("\nProbability masses:");
            let degree_p1 =
                if let Some(range) = result.var_supports[program.result].finite_nonempty_range() {
                    *range.end() as usize + 1
                } else {
                    50
                };
            let mut inputs = vec![TaylorPoly::one(); result.var_supports.num_vars()];
            inputs[program.result.id()] = TaylorPoly::var_at_zero(Var(0), degree_p1);
            let expansion = bound.eval_taylor::<Rational>(&inputs);
            for i in 0..degree_p1 {
                let prob = expansion.coefficient(&[i]).round_to_f64();
                println!("p({i}) <= {prob}");
            }
            println!("\nMoments:");
            let mut inputs = vec![TaylorPoly::one(); result.var_supports.num_vars()];
            inputs[program.result.id()] = TaylorPoly::var_at_zero(Var(0), 5).exp();
            let expansion = bound.eval_taylor::<Rational>(&inputs);
            let mut factorial = Rational::one();
            for i in 0..5 {
                let moment = (expansion.coefficient(&[i]) * factorial.clone()).round_to_f64();
                println!("{i}-th (raw) moment <= {moment}");
                factorial *= Rational::from_int(i + 1);
            }
        }
        Err(e) => match e {
            SolverError::Timeout => {
                println!("Solver timeout");
            }
            SolverError::Infeasible => {
                println!("Solver proved that there is no bound of the required form");
            }
        },
    }
    println!("Total time: {:?}", start.elapsed());
    Ok(())
}
