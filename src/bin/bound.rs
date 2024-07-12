#![warn(clippy::pedantic)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use tool::bounds::ctx::BoundCtx;
use tool::bounds::gradient_descent::{Adam, AdamBarrier, GradientDescent};
use tool::bounds::ipopt::Ipopt;
use tool::bounds::optimizer::{LinearProgrammingOptimizer, Optimizer as _, Z3Optimizer};
use tool::bounds::solver::{ConstraintProblem, Solver as _, SolverError, Z3Solver};
use tool::interval::Interval;
use tool::multivariate_taylor::TaylorPoly;
use tool::number::{Rational, F64};
use tool::parser;
use tool::ppl::{Program, Var};
use tool::semantics::support::VarSupport;
use tool::semantics::Transformer;

use clap::{Parser, ValueEnum};
use ndarray::Axis;
use num_traits::{One, Zero};

#[derive(Clone, ValueEnum)]
enum Solver {
    Z3,
    #[value(name = "gd")]
    GradientDescent,
    AdamBarrier,
    Ipopt,
}

#[derive(Clone, ValueEnum)]
enum Optimizer {
    Z3,
    #[value(name = "gd")]
    GradientDescent,
    Adam,
    AdamBarrier,
    Ipopt,
}

#[derive(Clone, ValueEnum)]
enum Objective {
    Total,
    #[value(name = "ev")]
    ExpectedValue,
    Tail,
    Balance,
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
    /// Disable normalization of the distribution
    #[arg(long)]
    no_normalize: bool,
    #[arg(short = 'd', long, default_value = "1")]
    /// The minimum degree of the loop invariant polynomial
    min_degree: usize,
    #[arg(short = 'u', long, default_value = "8")]
    /// The default number of loop unrollings
    unroll: usize,
    /// The limit for the probability masses to be computed
    #[arg(short, long)]
    pub limit: Option<usize>,
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

pub fn main() -> std::io::Result<ExitCode> {
    let args = CliArgs::parse();
    let contents = std::fs::read_to_string(&args.file_name)?;
    let program = parser::parse_program(&contents);
    if args.print_program {
        println!("Parsed program:\n{program}\n");
    }
    run_program(&program, &args)
}

fn run_program(program: &Program, args: &CliArgs) -> std::io::Result<ExitCode> {
    let start = Instant::now();
    let mut ctx = BoundCtx::new()
        .with_verbose(args.verbose)
        .with_min_degree(args.min_degree)
        .with_unroll(args.unroll)
        .with_evt(args.evt)
        .with_do_while_transform(!args.keep_while);
    let mut result = ctx.semantics(program);
    println!(
        "Generated {} constraints with {} symbolic variables.",
        ctx.constraints().len(),
        ctx.sym_var_count()
    );
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
        for (v, (lo, hi)) in ctx.sym_var_bounds().iter().enumerate() {
            println!("  x{v} ∈ [{lo}, {hi})");
        }
        for constraint in ctx.constraints() {
            println!("  {constraint}");
        }
        println!("Polynomial constraints:");
        for constraint in ctx.constraints() {
            println!("  {}", constraint.to_poly());
        }
    }
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
    println!(
        "Constraint generation time: {:.5}s",
        time_constraint_gen.as_secs_f64()
    );
    println!("Solving constraints...");
    let start_solver = Instant::now();
    let timeout = Duration::from_millis(args.timeout);
    let problem = ConstraintProblem {
        var_count: ctx.sym_var_count(),
        geom_vars: ctx.geom_vars().to_owned(),
        factor_vars: ctx.factor_vars().to_owned(),
        coefficient_vars: ctx.coefficient_vars().to_owned(),
        var_bounds: ctx.sym_var_bounds().to_owned(),
        constraints: ctx.constraints().to_owned(),
    };
    let init_solution = match args.solver {
        Solver::Z3 => Z3Solver.solve(&problem, timeout),
        Solver::GradientDescent => GradientDescent::default()
            .with_verbose(args.verbose)
            .solve(&problem, timeout),
        Solver::AdamBarrier => AdamBarrier::default()
            .with_verbose(args.verbose)
            .solve(&problem, timeout),
        Solver::Ipopt => Ipopt::default()
            .with_verbose(args.verbose)
            .solve(&problem, timeout),
    };
    let solver_time = start_solver.elapsed();
    println!("Solver time: {:.5}s", solver_time.as_secs_f64());
    let exit_code = match init_solution {
        Ok(solution) => {
            let objective = match args.objective {
                Objective::Total => result.upper.total_mass(),
                Objective::ExpectedValue => result.upper.expected_value(program.result),
                Objective::Tail => result.upper.tail_objective(program.result),
                Objective::Balance => {
                    result.upper.total_mass() * result.upper.tail_objective(program.result).pow(4)
                }
            };
            println!("Optimizing solution...");
            let optimized_solution = if let Some(optimizer) = &args.optimizer {
                let start_optimizer = Instant::now();
                let optimized_solution = match optimizer {
                    Optimizer::Z3 => {
                        Z3Optimizer.optimize(&problem, &objective, solution.clone(), timeout)
                    }
                    Optimizer::GradientDescent => GradientDescent::default()
                        .with_verbose(args.verbose)
                        .optimize(&problem, &objective, solution, timeout),
                    Optimizer::Adam => Adam::default()
                        .with_verbose(args.verbose)
                        .optimize(&problem, &objective, solution, timeout),
                    Optimizer::AdamBarrier => AdamBarrier::default()
                        .with_verbose(args.verbose)
                        .optimize(&problem, &objective, solution, timeout),
                    Optimizer::Ipopt => Ipopt::default()
                        .with_verbose(args.verbose)
                        .optimize(&problem, &objective, solution, timeout),
                };
                let optimizer_time = start_optimizer.elapsed();
                println!("Optimizer time: {:.6}", optimizer_time.as_secs_f64());
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
            result.upper = result.upper.resolve(&optimized_solution);
            if args.verbose {
                println!("\nFinal (unnormalized) bound:\n");
                println!("{result}");
            }

            for v in 0..result.var_supports.num_vars() {
                if Var(v) != program.result {
                    result = result.marginalize(Var(v));
                }
            }
            println!("\nMarginalized bound:");
            let ax = Axis(program.result.id());
            let upper_len = result.upper.masses.len_of(ax);
            let lower_len = result.lower.masses.len_of(ax);
            let thresh = lower_len.max(upper_len - 1);
            let decay = result.upper.geo_params[program.result.id()]
                .extract_constant()
                .unwrap()
                .rat();
            for i in 0..thresh {
                let lo = if i < lower_len {
                    result
                        .lower
                        .masses
                        .index_axis(ax, i)
                        .first()
                        .unwrap()
                        .clone()
                } else {
                    Rational::zero()
                };
                let hi = if i < upper_len {
                    result
                        .upper
                        .masses
                        .index_axis(ax, i)
                        .first()
                        .unwrap()
                        .extract_constant()
                        .unwrap()
                        .rat()
                } else {
                    result
                        .upper
                        .masses
                        .index_axis(ax, upper_len - 1)
                        .first()
                        .unwrap()
                        .extract_constant()
                        .unwrap()
                        .rat()
                        * decay.pow((i - upper_len).try_into().unwrap())
                };
                println!(
                    "{i}: [{}, {}]",
                    F64::from(lo.round_to_f64()),
                    F64::from(hi.round_to_f64())
                );
            }
            let thresh_hi = result
                .upper
                .masses
                .index_axis(ax, upper_len - 1)
                .first()
                .unwrap()
                .extract_constant()
                .unwrap()
                .rat()
                * decay.pow((thresh + 1 - upper_len).try_into().unwrap());
            println!(
                "n >= {thresh}: [0, {} * {}^(n - {thresh})]",
                F64::from(thresh_hi.round_to_f64()),
                F64::from(decay.round_to_f64()),
            );

            // Compute bounds on the normalizing constant:
            let (norm_lo, norm_hi) = if !args.no_normalize && program.uses_observe() {
                let total_lo = result.lower.total_mass();
                let total_hi = result.upper.total_mass().extract_constant().unwrap().rat();
                let total_hi = if total_hi > Rational::one() {
                    Rational::one()
                } else {
                    total_hi
                };
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
            let limit =
                if let Some(range) = result.var_supports[program.result].finite_nonempty_range() {
                    *range.end() as usize + 1
                } else {
                    args.limit.unwrap_or(50)
                };
            let limit = limit.max(2);
            let mut inputs = vec![TaylorPoly::one(); result.var_supports.num_vars()];
            inputs[program.result.id()] = TaylorPoly::var_at_zero(Var(0), limit);
            let lower_probs = result.lower.probs(program.result);
            let expansion = result.upper.eval_taylor::<Rational>(&inputs);
            for i in 0..limit {
                let lo = lower_probs.get(i).unwrap_or(&Rational::zero()).clone() / norm_hi.clone();
                let mut hi = expansion.coefficient(&[i]) / norm_lo.clone();
                if hi > Rational::one() {
                    hi = Rational::one();
                }
                let prob = Interval::exact(lo, hi);
                println!("p({i}) {}", in_iv(&prob));
            }
            if decay.is_zero() {
                let from = if thresh_hi.is_zero() {
                    thresh
                } else {
                    thresh + 1
                };
                println!("Asymptotics: p(n) = 0 for n >= {from}");
            } else {
                let factor =
                    thresh_hi / norm_lo.clone() * decay.pow(-(i32::try_from(thresh).unwrap()));
                println!(
                    "\nAsymptotics: p(n) <= {} * {}^n for n >= {}",
                    F64::from(factor.round_to_f64()),
                    F64::from(decay.round_to_f64()),
                    thresh
                );
            }

            println!("\nMoments:");
            let lower_moments = result.lower.moments(program.result, 5);
            let mut inputs = vec![TaylorPoly::one(); result.var_supports.num_vars()];
            inputs[program.result.id()] = TaylorPoly::var_at_zero(Var(0), 5).exp();
            let expansion = result.upper.eval_taylor::<Rational>(&inputs);
            let mut factorial = Rational::one();
            for i in 0..5 {
                let lo = lower_moments[i].clone() / norm_hi.clone();
                let hi = expansion.coefficient(&[i]) / norm_lo.clone();
                let moment = Interval::exact(lo, hi);
                println!("{i}-th (raw) moment {}", in_iv(&moment));
                factorial *= Rational::from_int(i + 1);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            match e {
                SolverError::Timeout => {
                    eprintln!("Solver failed: timeout");
                }
                SolverError::Infeasible => {
                    eprintln!("Solver failed: it claims the problem is infeasible");
                }
                SolverError::Other => {
                    eprintln!("Solver failed: unknown reason");
                }
            }
            ExitCode::FAILURE
        }
    };
    println!("Total time: {:.5}s", start.elapsed().as_secs_f64());
    Ok(exit_code)
}

fn in_iv(iv: &Interval<Rational>) -> String {
    if iv.lo == iv.hi {
        format!("= {}", F64::from(iv.lo.round_to_f64()))
    } else {
        format!(
            "∈ [{}, {}]",
            F64::from(iv.lo.round_down()),
            F64::from(iv.hi.round_up())
        )
    }
}
