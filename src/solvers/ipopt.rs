use std::time::Duration;

use crate::{numbers::Rational, sym_expr::SymConstraint};

use super::{
    optimizer::Optimizer,
    solver::{ConstraintProblem, Solver, SolverError},
};

use descent::{
    expr::{dynam::Expr, Var},
    model::{Model, SolutionStatus},
};
use descent_ipopt::IpoptModel;
use rustc_hash::FxHashMap;

pub struct Ipopt {
    verbose: bool,
    tol: f64,
}

impl Ipopt {
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn construct_model(
        &self,
        problem: &ConstraintProblem,
        init: Option<&[Rational]>,
    ) -> (Vec<Var>, IpoptModel) {
        let cache = &mut FxHashMap::default();
        let mut model = IpoptModel::new();
        model.set_num_option("constr_viol_tol", self.tol);
        let mut vars = Vec::new();
        for (v, (lo, hi)) in problem.var_bounds.iter().enumerate() {
            let (lo, hi) = (lo.round_to_f64(), hi.round_to_f64());
            let start = if let Some(init) = init {
                init[v].round_to_f64()
            } else if hi.is_finite() {
                (lo + hi) / 2.0
            } else {
                2.0 * lo
            };
            vars.push(model.add_var(lo, hi, start));
        }
        for constraint in &problem.constraints {
            if let Some((expr, lo, hi)) = to_ipopt_constraint(constraint, &vars, self.tol, cache) {
                model.add_con(expr, lo, hi);
            }
        }
        model.set_obj(problem.objective.to_ipopt_expr(&vars, cache));
        (vars, model)
    }

    pub fn solve(
        vars: &[Var],
        model: &mut IpoptModel,
    ) -> Result<(SolutionStatus, Vec<Rational>), String> {
        let (status, solution) = model.solve().map_err(|e| e.to_string())?;
        let point = vars.iter().map(|v| solution.var(*v).into()).collect();
        Ok((status, point))
    }
}

impl Default for Ipopt {
    fn default() -> Self {
        Ipopt {
            verbose: false,
            tol: 1e-7,
        }
    }
}

impl Solver for Ipopt {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        _timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError> {
        let (vars, mut model) = self.construct_model(problem, None);
        match Self::solve(&vars, &mut model) {
            Ok((status, solution)) => match status {
                SolutionStatus::Solved => {
                    if self.verbose {
                        println!(
                            "IPOPT found the following solution: {:?}",
                            solution
                                .iter()
                                .map(Rational::round_to_f64)
                                .collect::<Vec<_>>()
                        );
                    }
                    if problem.holds_exact(&solution) {
                        println!("Solution satisfies all constraints.");
                        Ok(solution)
                    } else {
                        println!("Solution does not satisfy all constraints (rounding errors?).");
                        Err(SolverError::Timeout)
                    }
                }
                SolutionStatus::Infeasible => {
                    println!("IPOPT found the problem infeasible.");
                    Err(SolverError::Infeasible)
                }
                SolutionStatus::Other => {
                    println!("IPOPT failed to solve the problem.");
                    Err(SolverError::Other)
                }
            },
            Err(e) => {
                println!("IPOPT failed: {e}");
                Err(SolverError::Other)
            }
        }
    }
}

impl Optimizer for Ipopt {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        init: Vec<Rational>,
        _timeout: Duration,
    ) -> Vec<Rational> {
        let init_obj = problem.objective.eval_exact(&init, &mut Default::default());
        let (vars, mut model) = self.construct_model(problem, Some(&init));
        match Self::solve(&vars, &mut model) {
            Ok((status, solution)) => {
                let cache = &mut FxHashMap::default();
                let obj_value = problem.objective.eval_exact(&solution, cache);
                let holds_exact = problem.holds_exact_with(&solution, cache);
                match status {
                    SolutionStatus::Solved => {
                        println!("IPOPT found the following solution:");
                        println!(
                            "Objective: {} at {:?}",
                            obj_value.round_to_f64(),
                            solution
                                .iter()
                                .map(Rational::round_to_f64)
                                .collect::<Vec<_>>()
                        );
                        if holds_exact {
                            println!("The solution satisfies all constraints.");
                            solution
                        } else {
                            println!("The solution does not satisfy all constraints (rounding errors?); returning initial solution.");
                            init
                        }
                    }
                    SolutionStatus::Infeasible => {
                        println!("IPOPT found the problem infeasible; returning initial solution.");
                        init
                    }
                    SolutionStatus::Other => {
                        if holds_exact && obj_value <= init_obj {
                            println!("IPOPT failed to find an optimal solution; continuing with nonoptimal solution.");
                            solution
                        } else {
                            println!(
                                "IPOPT failed to solve the problem; returning initial solution."
                            );
                            init
                        }
                    }
                }
            }
            Err(e) => {
                println!("IPOPT failed: {e}\nReturning initial solution.");
                init
            }
        }
    }
}

fn to_ipopt_constraint(
    constraint: &SymConstraint,
    vars: &[Var],
    tol: f64,
    cache: &mut FxHashMap<usize, Expr>,
) -> Option<(Expr, f64, f64)> {
    match constraint {
        SymConstraint::Eq(lhs, rhs) => {
            let expr = lhs.to_ipopt_expr(vars, cache) - rhs.to_ipopt_expr(vars, cache);
            Some((expr, 0.0, 0.0))
        }
        SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
            let expr = lhs.to_ipopt_expr(vars, cache) - rhs.to_ipopt_expr(vars, cache);
            Some((expr, f64::NEG_INFINITY, -2.0 * tol))
        }
        SymConstraint::Or(_) => None,
    }
}
