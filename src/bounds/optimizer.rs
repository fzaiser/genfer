use std::time::Duration;

use good_lp::{variable, ProblemVariables, Solution, SolverModel};

use crate::bounds::util::{f64_to_z3, z3_real_to_f64};

use super::{solver::ConstraintProblem, sym_expr::SymExpr};

pub trait Optimizer {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        objective: &SymExpr<f64>,
        init: Vec<f64>,
        timeout: Duration,
    ) -> Vec<f64>;
}

pub struct Z3Optimizer;

impl Optimizer for Z3Optimizer {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        objective: &SymExpr<f64>,
        init: Vec<f64>,
        timeout: Duration,
    ) -> Vec<f64> {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for constraint in &problem.constraints {
            solver.assert(&constraint.to_z3(&ctx, &f64_to_z3));
        }
        solver.push();
        let mut best = init;
        let mut obj_lo = 0.0;
        let mut obj_hi = objective.eval(&best);
        while obj_hi - obj_lo > 0.1 * obj_hi {
            println!("Objective bound: [{obj_lo}, {obj_hi}]");
            solver.pop(1);
            solver.push();
            let mid = (obj_lo + obj_hi) / 2.0;
            println!("Trying objective bound: {mid}");
            solver.assert(
                &objective
                    .to_z3(&ctx, &f64_to_z3)
                    .le(&SymExpr::from(mid).to_z3(&ctx, &f64_to_z3)),
            );
            match solver.check() {
                z3::SatResult::Sat => {
                    println!("Solution found for these objective bounds.");
                }
                z3::SatResult::Unsat => {
                    println!("No solution for these objective bounds.");
                    obj_lo = mid;
                    continue;
                }
                z3::SatResult::Unknown => {
                    println!("Solver responded 'unknown' while optimizing the objective. Aborting optimization.");
                    println!(
                        "Reason for unknown: {}",
                        solver.get_reason_unknown().unwrap_or("none".to_owned())
                    );
                    break;
                }
            }
            if let Some(model) = solver.get_model() {
                let obj_val = z3_real_to_f64(
                    &model
                        .eval(&objective.to_z3(&ctx, &f64_to_z3), false)
                        .unwrap(),
                )
                .unwrap();
                println!("Total mass (objective): {obj_val}");
                assert!(obj_val < obj_hi);
                obj_hi = obj_val;
                best = (0..problem.var_count)
                    .map(|var| {
                        let val = model
                            .eval(&SymExpr::var(var).to_z3(&ctx, &f64_to_z3), false)
                            .unwrap();
                        z3_real_to_f64(&val)
                            .unwrap_or_else(|| panic!("{val} cannot be converted to f64"))
                    })
                    .collect::<Vec<_>>();
            } else {
                println!("Solver returned SAT but no model. Aborting.");
                break;
            }
        }
        best
    }
}

pub struct LinearProgrammingOptimizer;

impl Optimizer for LinearProgrammingOptimizer {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        objective: &SymExpr<f64>,
        init: Vec<f64>,
        _timeout: Duration,
    ) -> Vec<f64> {
        optimize_linear_parts(problem, objective, init.clone()).unwrap_or_else(|| {
            println!("LP solver failed to optimize linear parts. Returning initial solution.");
            init
        })
    }
}

pub fn optimize_linear_parts(
    problem: &ConstraintProblem,
    objective: &SymExpr<f64>,
    init: Vec<f64>,
) -> Option<Vec<f64>> {
    let mut replacements = (0..problem.var_count).map(SymExpr::var).collect::<Vec<_>>();
    for v in problem.geom_vars.iter().chain(problem.factor_vars.iter()) {
        replacements[*v] = SymExpr::Constant(init[*v]);
    }
    let objective = objective.substitute(&replacements);
    let constraints = problem
        .constraints
        .iter()
        .map(|c| c.substitute(&replacements))
        .collect::<Vec<_>>();
    let linear_constraints = constraints
        .iter()
        .map(|constraint| {
            constraint
                .extract_linear()
                .unwrap_or_else(|| {
                    panic!("Constraint is not linear in the program variables: {constraint}")
                })
                .normalize()
                .tighten(1e-6)
        })
        .collect::<Vec<_>>();
    let mut lp = ProblemVariables::new();
    let mut var_list = Vec::new();
    for replacement in replacements {
        match replacement {
            SymExpr::Variable(_) => {
                var_list.push(lp.add_variable());
            }
            SymExpr::Constant(c) => {
                var_list.push(lp.add(variable().min(c).max(c)));
            }
            _ => unreachable!(),
        }
    }
    let linear_objective = objective
        .extract_linear()
        .unwrap_or_else(|| panic!("Objective is not linear in the program variables: {objective}"))
        .to_lp_expr(&var_list, &|v| *v);
    let mut lp = lp.minimise(linear_objective).using(good_lp::default_solver);
    for constraint in &linear_constraints {
        lp.add_constraint(constraint.to_lp_constraint(&var_list, &|v| *v));
    }
    let solution = match lp.solve() {
        Ok(solution) => solution,
        Err(good_lp::ResolutionError::Unbounded) => {
            println!("Optimal solution is unbounded.");
            return None;
        }
        Err(good_lp::ResolutionError::Infeasible) => {
            println!("LP solver found the problem infeasible.");
            return None;
        }
        Err(good_lp::ResolutionError::Other(msg)) => {
            todo!("Other error: {msg}");
        }
        Err(good_lp::ResolutionError::Str(msg)) => {
            todo!("Error: {msg}");
        }
    };
    let solution = var_list
        .iter()
        .map(|v| solution.value(*v))
        .collect::<Vec<_>>();
    let solution = if problem.holds_exact(&solution) {
        solution
    } else {
        println!("Solution by LP solver does not satisfy the constraints.");
        return None;
    };
    println!(
        "Best objective: {} at {:?}",
        objective.eval(&solution),
        solution
    );
    Some(solution)
}
