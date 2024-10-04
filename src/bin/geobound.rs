#![warn(clippy::pedantic)]
#![expect(clippy::needless_range_loop)]
#![expect(clippy::cast_possible_truncation)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use tool::bound::GeometricBound;
use tool::interval::Interval;
use tool::numbers::{Rational, F64};
use tool::parser;
use tool::ppl::{Program, Var};
use tool::semantics::geometric::GeometricBoundSemantics;
use tool::semantics::support::VarSupport;
use tool::semantics::Transformer;
use tool::solvers::gradient_descent::{Adam, AdamBarrier, GradientDescent};
use tool::solvers::ipopt::Ipopt;
use tool::solvers::optimizer::{
    optimize_linear_parts, LinearProgrammingOptimizer, Optimizer as _, Z3Optimizer,
};
use tool::solvers::solver::{ConstraintProblem, Solver as _, SolverError, Z3Solver};

use clap::{Parser, ValueEnum};
use ndarray::Axis;
use num_traits::{One, Zero};
use tool::sym_expr::SymExpr;

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
    Linear,
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Optimizer::Z3 => write!(f, "Z3"),
            Optimizer::GradientDescent => write!(f, "Gradient Descent"),
            Optimizer::Adam => write!(f, "ADAM"),
            Optimizer::AdamBarrier => write!(f, "ADAM with Barrier Method"),
            Optimizer::Ipopt => write!(f, "IPOPT"),
            Optimizer::Linear => write!(f, "Linear Optimization"),
        }
    }
}

#[derive(Copy, Clone, ValueEnum)]
enum Objective {
    Total,
    #[value(name = "ev")]
    ExpectedValue,
    Tail,
    Balance,
}

#[expect(clippy::struct_excessive_bools)]
#[derive(Clone, Parser)]
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
    #[arg(long, default_value = "ipopt")]
    solver: Solver,
    /// The optimizer to use
    #[arg(long, default_values = ["ipopt", "adam-barrier", "linear"])]
    optimizer: Vec<Optimizer>,
    /// The optimization objective
    #[arg(long)]
    objective: Option<Objective>,
    #[arg(long)]
    evt: bool,
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
    Ok(run_program(&program, &args))
}

fn run_program(program: &Program, args: &CliArgs) -> ExitCode {
    let start = Instant::now();
    let timeout = Duration::from_millis(args.timeout);
    let (problem, bound, init_solution) = compute_constraints_solution(args, program, timeout);
    let exit_code = match init_solution {
        Ok(solution) => continue_with_solution(args, bound, program, &problem, solution),
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
    exit_code
}

fn compute_constraints_solution(
    args: &CliArgs,
    program: &Program,
    timeout: Duration,
) -> (
    ConstraintProblem,
    GeometricBound,
    Result<Vec<Rational>, SolverError>,
) {
    if args.unroll != 0 {
        println!("Solving simplified problem with unroll limit set to 0 first...");
        let modified_args = CliArgs {
            unroll: 1,
            objective: None,
            ..args.clone()
        };
        let (simple_problem, _) = generate_constraints(&modified_args, program);
        let simple_solution = solve_constraints(&modified_args, &simple_problem, timeout);
        if let Ok(simple_solution) = simple_solution {
            println!("Optimizing solution to the simplified problem...");
            let modified_args = CliArgs {
                objective: args.objective,
                ..args.clone()
            };
            let simple_solution = optimize_solution(
                &modified_args,
                &simple_problem,
                simple_solution.clone(),
                timeout,
            );
            println!("Extending solution to the original problem...");
            let (problem, bound) = generate_constraints(args, program);
            // The nonlinear variables can be reused from the simplified problem:
            let mut solution = vec![Rational::zero(); problem.var_count];
            for (simple_var, var) in simple_problem.factor_vars.iter().zip(&problem.factor_vars) {
                solution[*var] = simple_solution[*simple_var].clone();
            }
            for (simple_var, var) in simple_problem.decay_vars.iter().zip(&problem.decay_vars) {
                solution[*var] = simple_solution[*simple_var].clone();
            }
            // Solve the linear variables:
            if let Some(solution) = optimize_linear_parts(&problem, solution) {
                return (problem, bound, Ok(solution));
            }
        }
        println!("Solving simplified problem (unrolling limit set to 0) failed.");
        println!("Continuing with the original problem.");
    }
    let (problem, bound) = generate_constraints(args, program);
    let init_solution = solve_constraints(args, &problem, timeout);
    (problem, bound, init_solution)
}

fn generate_constraints(args: &CliArgs, program: &Program) -> (ConstraintProblem, GeometricBound) {
    let start = Instant::now();
    let mut ctx = GeometricBoundSemantics::new()
        .with_verbose(args.verbose)
        .with_min_degree(args.min_degree)
        .with_unroll(args.unroll)
        .with_evt(args.evt)
        .with_do_while_transform(!args.keep_while);
    let bound = ctx.semantics(program);
    println!(
        "Generated {} constraints with {} symbolic variables.",
        ctx.constraints().len(),
        ctx.sym_var_count()
    );
    if args.verbose {
        match &bound.var_supports {
            VarSupport::Empty(_) => println!("Support: empty"),
            VarSupport::Prod(supports) => {
                for (v, support) in supports.iter().enumerate() {
                    println!("Support of {}: {support}", Var(v));
                }
            }
        }
        println!("Bound result:");
        println!("{bound}");
        println!("Constraints:");
        for (v, (lo, hi)) in ctx.sym_var_bounds().iter().enumerate() {
            println!("  x{v} ∈ [{lo}, {hi})");
        }
        for constraint in ctx.constraints() {
            println!("  {constraint}");
        }
    }
    if let Some(path) = &args.smt {
        println!("Writing SMT file to {path:?}...");
        let mut out = std::fs::File::create(path).unwrap();
        ctx.output_smt(&mut out).unwrap();
    }
    if let Some(path) = &args.qepcad {
        println!("Writing QEPCAD commands to {path:?}...");
        let mut out = std::fs::File::create(path).unwrap();
        ctx.output_qepcad(&mut out).unwrap();
    }
    let objective = objective_function(&bound, program.result, args.objective);
    let mut problem = ConstraintProblem {
        var_count: ctx.sym_var_count(),
        decay_vars: ctx.geom_vars().to_owned(),
        factor_vars: ctx.factor_vars().to_owned(),
        block_vars: ctx.block_vars().to_owned(),
        var_bounds: ctx.sym_var_bounds().to_owned(),
        constraints: ctx.constraints().to_owned(),
        objective,
    };
    problem.preprocess();
    let time_constraint_gen = start.elapsed();
    println!(
        "Constraint generation time: {:.5}s",
        time_constraint_gen.as_secs_f64()
    );
    (problem, bound)
}

fn solve_constraints(
    args: &CliArgs,
    problem: &ConstraintProblem,
    timeout: Duration,
) -> Result<Vec<Rational>, SolverError> {
    println!("Solving constraints...");
    let start_solver = Instant::now();
    let solution = match args.solver {
        Solver::Z3 => Z3Solver.solve(problem, timeout),
        Solver::GradientDescent => GradientDescent::default()
            .with_verbose(args.verbose)
            .solve(problem, timeout),
        Solver::AdamBarrier => AdamBarrier::default()
            .with_verbose(args.verbose)
            .solve(problem, timeout),
        Solver::Ipopt => Ipopt::default()
            .with_verbose(args.verbose)
            .solve(problem, timeout),
    };
    let solver_time = start_solver.elapsed();
    println!("Solver time: {:.5}s", solver_time.as_secs_f64());
    solution
}

fn continue_with_solution(
    args: &CliArgs,
    mut bound: GeometricBound,
    program: &Program,
    problem: &ConstraintProblem,
    solution: Vec<Rational>,
) -> ExitCode {
    let timeout = Duration::from_millis(args.timeout);
    let optimized_solution = optimize_solution(args, problem, solution, timeout);
    bound.upper = bound.upper.resolve(&optimized_solution);
    // TODO: bound the probability masses by 1 (or even by the residual mass bound)
    if args.verbose {
        println!("\nFinal (unnormalized) bound:\n");
        println!("{bound}");
    }
    let marginal = bound.marginal(program.result);
    let (thresh, decay, thresh_hi) = output_unnormalized(&marginal, program.result);
    let (norm_lo, norm_hi) = bound_normalization_constant(args, program, &marginal);
    output_probabilities(&marginal, program.result, args.limit, &norm_lo, &norm_hi);
    output_tail_asymptotics(&decay, thresh_hi, thresh, &norm_lo);
    output_moments(&marginal, program.result, &norm_lo, &norm_hi);
    ExitCode::SUCCESS
}

fn optimize_solution(
    args: &CliArgs,
    problem: &ConstraintProblem,
    mut solution: Vec<Rational>,
    timeout: Duration,
) -> Vec<Rational> {
    if problem.objective.is_zero() {
        println!(
            "SKIPPING optimization because no objective is set. Resulting bounds will be BAD."
        );
        return solution;
    };
    let start_optimizer = Instant::now();
    println!("Optimizing solution...");
    let mut objective = problem
        .objective
        .eval_exact(&solution, &mut Default::default());
    for (i, optimizer) in args.optimizer.iter().enumerate() {
        println!("Optimization step {}: {optimizer}", i + 1);
        let cur_sol = solution.clone();
        let start = Instant::now();
        let optimized_solution = match optimizer {
            Optimizer::Z3 => Z3Optimizer.optimize(problem, cur_sol, timeout),
            Optimizer::GradientDescent => GradientDescent::default()
                .with_verbose(args.verbose)
                .optimize(problem, cur_sol, timeout),
            Optimizer::Adam => Adam::default()
                .with_verbose(args.verbose)
                .optimize(problem, cur_sol, timeout),
            Optimizer::AdamBarrier => AdamBarrier::default()
                .with_verbose(args.verbose)
                .optimize(problem, cur_sol, timeout),
            Optimizer::Ipopt => Ipopt::default()
                .with_verbose(args.verbose)
                .optimize(problem, cur_sol, timeout),
            Optimizer::Linear => LinearProgrammingOptimizer.optimize(problem, cur_sol, timeout),
        };
        println!(
            "Optimizer ({optimizer}) time: {:.6} s",
            start.elapsed().as_secs_f64()
        );
        if let Some(optimized_objective) = problem.objective_if_holds_exactly(&optimized_solution) {
            if optimized_objective <= objective {
                solution = optimized_solution;
                objective = optimized_objective;
            } else {
                println!("Optimization step failed (worse objective). Continuing with the previous solution.");
            }
        } else {
            println!("Optimization step failed (constraint violation). Continuing with the previous solution.");
        }
    }
    let optimizer_time = start_optimizer.elapsed();
    println!("Optimization time: {:.6}", optimizer_time.as_secs_f64());
    solution
}

fn output_unnormalized(bound: &GeometricBound, var: Var) -> (usize, Rational, Rational) {
    println!("\nUnnormalized bound:");
    let ax = Axis(var.id());
    let upper_len = bound.upper.block.len_of(ax);
    let lower_len = bound.lower.masses.len_of(ax);
    let thresh = lower_len.max(upper_len - 1);
    let decay = bound.upper.decays[var.id()]
        .extract_constant()
        .unwrap()
        .rat();
    let lower_probs = bound.lower.probs(var);
    let upper_probs = bound.upper.probs_exact(var, thresh + 1);
    for i in 0..thresh {
        let lo = if i < lower_len {
            lower_probs[i].clone()
        } else {
            Rational::zero()
        };
        let hi = upper_probs[i].clone();
        println!(
            "{i}: [{}, {}]",
            F64::from(lo.round_to_f64()),
            F64::from(hi.round_to_f64())
        );
    }
    let thresh_hi = upper_probs[thresh].clone();
    println!(
        "n >= {thresh}: [0, {} * {}^(n - {thresh})]",
        F64::from(thresh_hi.round_to_f64()),
        F64::from(decay.round_to_f64()),
    );
    (thresh, decay, thresh_hi)
}

fn bound_normalization_constant(
    args: &CliArgs,
    program: &Program,
    bound: &GeometricBound,
) -> (Rational, Rational) {
    if args.no_normalize || !program.uses_observe() {
        (Rational::one(), Rational::one())
    } else {
        let total_lo = bound.lower.total_mass();
        let total_hi = bound.upper.total_mass().extract_constant().unwrap().rat();
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
    }
}

fn output_probabilities(
    bound: &GeometricBound,
    var: Var,
    limit: Option<usize>,
    norm_lo: &Rational,
    norm_hi: &Rational,
) {
    println!("\nProbability masses:");
    let limit = if let Some(range) = bound.var_supports[var].finite_nonempty_range() {
        *range.end() as usize + 1
    } else {
        limit.unwrap_or(50)
    };
    let limit = limit.max(2);
    let lower_probs = bound.lower.probs(var);
    let upper_probs = bound.upper.probs_exact(var, limit);
    for i in 0..limit {
        let lo = lower_probs.get(i).unwrap_or(&Rational::zero()).clone() / norm_hi.clone();
        let mut hi = upper_probs[i].clone() / norm_lo.clone();
        if hi > Rational::one() {
            hi = Rational::one();
        }
        let prob = Interval::exact(lo, hi);
        println!("p({i}) {}", in_iv(&prob));
    }
}

fn output_tail_asymptotics(
    decay: &Rational,
    thresh_hi: Rational,
    thresh: usize,
    norm_lo: &Rational,
) {
    if decay.is_zero() {
        let from = if thresh_hi.is_zero() {
            thresh
        } else {
            thresh + 1
        };
        println!("Asymptotics: p(n) = 0 for n >= {from}");
    } else {
        let factor = thresh_hi / norm_lo.clone() * decay.pow(-(i32::try_from(thresh).unwrap()));
        println!(
            "\nAsymptotics: p(n) <= {} * {}^n for n >= {}",
            F64::from(factor.round_to_f64()),
            F64::from(decay.round_to_f64()),
            thresh
        );
    }
}

fn output_moments(result: &GeometricBound, var: Var, norm_lo: &Rational, norm_hi: &Rational) {
    println!("\nMoments:");
    let lower_moments = result.lower.moments(var, 5);
    let upper_moments = result.upper.moments_exact(var, 5);
    let mut factorial = Rational::one();
    for i in 0..5 {
        let lo = lower_moments[i].clone() / norm_hi.clone();
        let hi = upper_moments[i].clone() / norm_lo.clone();
        let moment = Interval::exact(lo, hi);
        println!("{i}-th (raw) moment {}", in_iv(&moment));
        factorial *= Rational::from_int(i + 1);
    }
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

fn objective_function(
    bound: &GeometricBound,
    result_var: Var,
    objective: Option<Objective>,
) -> SymExpr {
    match objective {
        None => SymExpr::zero(),
        Some(Objective::Total) => bound.upper.total_mass(),
        Some(Objective::ExpectedValue) => bound.upper.expected_value(result_var),
        Some(Objective::Tail) => bound.upper.tail_objective(result_var),
        Some(Objective::Balance) => {
            bound.upper.total_mass() * bound.upper.tail_objective(result_var).pow(4)
        }
    }
}
