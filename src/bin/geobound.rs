#![warn(clippy::pedantic)]
#![expect(clippy::needless_range_loop)]
#![expect(clippy::cast_possible_truncation)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rustc_hash::FxHashMap;
use tool::bound::GeometricBound;
use tool::interval::Interval;
use tool::numbers::{Rational, F64};
use tool::parser;
use tool::ppl::{Program, Var};
use tool::semantics::geometric::GeometricBoundSemantics;
use tool::semantics::support::VarSupport;
use tool::semantics::Transformer;
use tool::solvers::adam::AdamBarrier;
use tool::solvers::ipopt::Ipopt;

use clap::{Parser, ValueEnum};
use ndarray::Axis;
use num_traits::{One, Zero};
use tool::solvers::linear::{optimize_linear_parts, LinearProgrammingOptimizer};
use tool::solvers::problem::ConstraintProblem;
use tool::solvers::z3::Z3Solver;
use tool::solvers::{Optimizer as _, Solver as _, SolverError};
use tool::sym_expr::SymExpr;

#[derive(Clone, ValueEnum)]
enum Solver {
    Z3,
    AdamBarrier,
    Ipopt,
}

#[derive(Clone, ValueEnum)]
enum Optimizer {
    AdamBarrier,
    Ipopt,
    Linear,
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
    #[arg(short = 'd', long, default_value = "1")]
    /// The minimum degree of the loop invariant polynomial
    min_degree: usize,
    #[arg(short = 'u', long, default_value = "8")]
    /// The default number of loop unrollings
    unroll: usize,
    /// The limit for the probability masses to be computed
    #[arg(short, long)]
    pub limit: Option<usize>,
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
    smtlib: Option<PathBuf>,
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
    if args.objective.is_none() {
        println!("WARNING: No optimization objective set. The resulting bounds will be BAD.");
    }
    Ok(run_program(&program, &args))
}

fn run_program(program: &Program, args: &CliArgs) -> ExitCode {
    let start = Instant::now();
    let (problem, bound, init_solution) = compute_constraints_solution(args, program);
    let exit_code = match init_solution {
        Ok(solution) => continue_with_solution(args, bound, program, &problem, solution),
        Err(e) => {
            match e {
                SolverError::Failed => {
                    eprintln!("Solver failed to find a solution.");
                }
                SolverError::ProvenInfeasible => {
                    eprintln!("The constraint problem is infeasible. No geometric bound exists.");
                }
                SolverError::MaybeInfeasible => {
                    eprintln!("Solver failed, indicating that the problem may be infeasible.");
                }
                SolverError::Other => {
                    eprintln!("Solver failed: unknown reason");
                }
            }
            eprintln!("Try varying the options (unrolling, invariant size, etc.).");
            ExitCode::FAILURE
        }
    };
    output_time("Total time", start, args.no_timing);
    exit_code
}

fn compute_constraints_solution(
    args: &CliArgs,
    program: &Program,
) -> (
    ConstraintProblem,
    GeometricBound,
    Result<Vec<Rational>, SolverError>,
) {
    println!("\nCONSTRAINT GENERATION:");
    let (problem, bound) = generate_constraints(args, program);
    if let Some(path) = &args.smtlib {
        println!("\nWriting SMT-LIB file to {path:?}...");
        let mut out = std::fs::File::create(path).unwrap();
        problem.output_smtlib(&mut out).unwrap();
    }
    if let Some(path) = &args.qepcad {
        println!("\nWriting QEPCAD commands to {path:?}...");
        let mut out = std::fs::File::create(path).unwrap();
        problem.output_qepcad(&mut out).unwrap();
    }
    if args.unroll != 0 {
        println!("\nSOLVING SIMPLIFIED PROBLEM:");
        println!("Solving without loop unrolling first...");
        let modified_args = CliArgs {
            unroll: 1,
            ..args.clone()
        };
        println!("\nCONSTRAINT GENERATION (simplified problem):");
        let (simple_problem, _) = generate_constraints(&modified_args, program);
        println!("\nSOLVING CONSTRAINTS (simplified problem):");
        let simple_solution = solve_constraints(&modified_args, &simple_problem);
        if let Ok(simple_solution) = simple_solution {
            println!("\nOPTIMIZATION (simplified problem):");
            let simple_solution =
                optimize_solution(&modified_args, &simple_problem, simple_solution.clone());
            println!("\nEXTENDING SOLUTION:");
            println!("Extending simplified solution to the original problem...");
            let start_extend = Instant::now();
            // The nonlinear variables can be reused from the simplified problem:
            let mut solution = vec![Rational::zero(); problem.var_count];
            for (simple_var, var) in simple_problem.factor_vars.iter().zip(&problem.factor_vars) {
                solution[*var] = simple_solution[*simple_var].clone();
            }
            for (simple_var, var) in simple_problem.decay_vars.iter().zip(&problem.decay_vars) {
                solution[*var] = simple_solution[*simple_var].clone();
            }
            let solution = optimize_linear_parts(&problem, solution);
            output_time("Extension time (LP solver)", start_extend, args.no_timing);
            // Solve the linear variables:
            if let Some(solution) = solution {
                return (problem, bound, Ok(solution));
            }
        }
        println!("Solving simplified problem (unrolling limit set to 0) failed.");
        println!("Continuing with the original problem.");

        println!("\nSOLVING CONSTRAINTS (original problem):");
    } else {
        println!("\nSOLVING CONSTRAINTS:");
    }
    let init_solution = solve_constraints(args, &problem);
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
    output_time("Constraint generation time", start, args.no_timing);
    (problem, bound)
}

fn solve_constraints(
    args: &CliArgs,
    problem: &ConstraintProblem,
) -> Result<Vec<Rational>, SolverError> {
    let problem = &ConstraintProblem {
        objective: SymExpr::zero(),
        ..problem.clone()
    };
    let start_solver = Instant::now();
    let solution = match args.solver {
        Solver::Z3 => Z3Solver.solve(problem),
        Solver::AdamBarrier => AdamBarrier::default()
            .with_verbose(args.verbose)
            .solve(problem),
        Solver::Ipopt => Ipopt::default().with_verbose(args.verbose).solve(problem),
    };
    output_time("Solver time", start_solver, args.no_timing);
    solution
}

fn continue_with_solution(
    args: &CliArgs,
    mut bound: GeometricBound,
    program: &Program,
    problem: &ConstraintProblem,
    solution: Vec<Rational>,
) -> ExitCode {
    println!("\nOPTIMIZATION:");
    let optimized_solution = optimize_solution(args, problem, solution);
    bound.upper = bound.upper.resolve(&optimized_solution);
    // TODO: bound the probability masses by 1 (or even by the residual mass bound)
    if args.verbose {
        println!("\nFinal (unnormalized) bound:\n");
        println!("{bound}");
    }
    let marginal = bound.marginal(program.result);
    println!("\nUNNORMALIZED BOUND:");
    let (thresh, decay, thresh_hi) = output_unnormalized(&marginal, program.result);
    println!("\nNORMALIZED BOUND:");
    let norm = bound_normalization_constant(program, &marginal);
    output_probabilities(&marginal, program.result, args.limit, &norm, "p");
    output_tail_asymptotics(&decay, &thresh_hi, thresh, &norm.lo, "p");
    output_moments(&marginal, program.result, &norm);
    ExitCode::SUCCESS
}

fn optimize_solution(
    args: &CliArgs,
    problem: &ConstraintProblem,
    mut solution: Vec<Rational>,
) -> Vec<Rational> {
    if args.objective.is_none() {
        println!("No optimization objective set. Skipping optimization.");
        return solution;
    };
    let start_optimizer = Instant::now();
    let mut objective = problem
        .objective
        .eval_exact(&solution, &mut FxHashMap::default());
    println!("Initial objective: {}", F64::from(objective.to_f64()));
    for (i, optimizer) in args.optimizer.iter().enumerate() {
        println!("\nOptimization step {}: {optimizer}", i + 1);
        let cur_sol = solution.clone();
        let start = Instant::now();
        let new_solution = match optimizer {
            Optimizer::AdamBarrier => AdamBarrier::default()
                .with_verbose(args.verbose)
                .optimize(problem, cur_sol),
            Optimizer::Ipopt => Ipopt::default()
                .with_verbose(args.verbose)
                .optimize(problem, cur_sol),
            Optimizer::Linear => LinearProgrammingOptimizer.optimize(problem, cur_sol),
        };
        output_time(
            &format!("Optimizer time ({optimizer})"),
            start,
            args.no_timing,
        );
        if let Some(new_objective) = problem.objective_if_holds_exactly(&new_solution) {
            if new_objective < objective {
                solution = new_solution;
                objective = new_objective;
                println!("Improved objective: {}", F64::from(objective.to_f64()));
            } else if new_objective == objective {
                println!("Unchanged objective: {}", F64::from(new_objective.to_f64()));
            } else {
                println!(
                    "Worse objective: {} (continuing with previous solution)",
                    F64::from(new_objective.to_f64())
                );
            }
        } else {
            println!("Optimized solution violates constraints (continuing with previous solution)");
        }
    }
    output_time("\nTotal optimization time", start_optimizer, args.no_timing);
    solution
}

fn output_unnormalized(bound: &GeometricBound, var: Var) -> (usize, Rational, Rational) {
    let ax = Axis(var.id());
    let upper_len = bound.upper.block.len_of(ax);
    let lower_len = bound.lower.masses.len_of(ax);
    let thresh = lower_len.max(upper_len - 1);
    let decay = bound.upper.decays[var.id()]
        .extract_constant()
        .unwrap()
        .rat();
    let upper_probs = bound.upper.probs_exact(var, thresh + 1);
    output_probabilities(bound, var, Some(thresh), &Interval::one(), "p'");
    let thresh_hi = upper_probs[thresh].clone();
    output_tail_asymptotics(&decay, &thresh_hi, thresh, &Rational::one(), "p'");
    (thresh, decay, thresh_hi)
}

fn bound_normalization_constant(program: &Program, bound: &GeometricBound) -> Interval<Rational> {
    if program.uses_observe() {
        let total_lo = bound.lower.total_mass();
        let total_hi = bound.upper.total_mass().extract_constant().unwrap().rat();
        let total_hi = if total_hi > Rational::one() {
            Rational::one()
        } else {
            total_hi
        };
        let total = Interval::exact(total_lo, total_hi);
        println!("Normalizing constant: Z {}", in_iv(&total));
        total
    } else {
        println!("Normalizing constant: Z = 1 (no observe statements).");
        Interval::one()
    }
}

fn output_probabilities(
    bound: &GeometricBound,
    var: Var,
    limit: Option<usize>,
    norm: &Interval<Rational>,
    p: &str,
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
        let lo = lower_probs.get(i).unwrap_or(&Rational::zero()).clone() / norm.hi.clone();
        let mut hi = upper_probs[i].clone() / norm.lo.clone();
        if hi > Rational::one() {
            hi = Rational::one();
        }
        let prob = Interval::exact(lo, hi);
        println!("{p}({i}) {}", in_iv(&prob));
    }
}

fn output_tail_asymptotics(
    decay: &Rational,
    thresh_hi: &Rational,
    thresh: usize,
    norm_lo: &Rational,
    p: &str,
) {
    if decay.is_zero() {
        let from = if thresh_hi.is_zero() {
            thresh
        } else {
            thresh + 1
        };
        println!("\nAsymptotics: {p}(n) = 0 for n >= {from}");
    } else {
        let factor =
            thresh_hi.clone() / norm_lo.clone() * decay.pow(-(i32::try_from(thresh).unwrap()));
        println!(
            "\nAsymptotics: {p}(n) <= {} * {}^n for n >= {}",
            F64::from(factor.to_f64_up()),
            F64::from(decay.to_f64_up()),
            thresh
        );
    }
}

fn output_moments(result: &GeometricBound, var: Var, norm: &Interval<Rational>) {
    println!("\nMoments:");
    let lower_moments = result.lower.moments(var, 5);
    let upper_moments = result.upper.moments_exact(var, 5);
    let mut factorial = Rational::one();
    for i in 0..5 {
        let lo = lower_moments[i].clone() / norm.hi.clone();
        let hi = upper_moments[i].clone() / norm.lo.clone();
        let moment = Interval::exact(lo, hi);
        println!("{i}-th (raw) moment {}", in_iv(&moment));
        factorial *= Rational::from(i as u64 + 1);
    }
}

fn in_iv(iv: &Interval<Rational>) -> String {
    if iv.lo == iv.hi {
        format!("= {}", F64::from(iv.lo.to_f64()))
    } else {
        format!(
            "∈ [{}, {}]",
            F64::from(iv.lo.to_f64_down()),
            F64::from(iv.hi.to_f64_up())
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

fn output_time(name: &str, start: Instant, no_timing: bool) {
    if !no_timing {
        println!("{}: {:.5} s", name, start.elapsed().as_secs_f64());
    }
}
