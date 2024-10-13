use crate::{numbers::Rational, sym_expr::SymConstraint};

use super::{problem::ConstraintProblem, Optimizer, Solver, SolverError};

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

    pub(crate) fn construct_model(
        &self,
        problem: &ConstraintProblem,
        init: Option<&[Rational]>,
    ) -> (Vec<Var>, IpoptModel) {
        let cache = &mut FxHashMap::default();
        let mut model = IpoptModel::new();
        model.set_num_option("constr_viol_tol", self.tol);
        let mut vars = Vec::new();
        for (v, (lo, hi)) in problem.var_bounds.iter().enumerate() {
            let (lo, hi) = (lo.to_f64(), hi.to_f64());
            let start = if let Some(init) = init {
                init[v].to_f64()
            } else if hi.is_finite() {
                (lo + hi) / 2.0
            } else {
                2.0 * lo
            };
            vars.push(model.add_var(lo, hi, start));
        }
        for constraint in &problem.constraints {
            let (expr, lo, hi) = to_ipopt_constraint(constraint, &vars, self.tol, cache);
            model.add_con(expr, lo, hi);
        }
        // Weirdly, adding this trivial constraint helps IPOPT find feasible solutions:
        model.add_con(Expr::from(0.0), 0.0, 0.0);
        model.set_obj(problem.objective.to_ipopt_expr(&vars, cache));
        (vars, model)
    }

    pub(crate) fn solve(
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
    fn solve(&mut self, problem: &ConstraintProblem) -> Result<Vec<Rational>, SolverError> {
        let (vars, mut model) = self.construct_model(problem, None);
        // Set objective to zero because we're just solving, not optimizing
        model.set_obj(Expr::from(0.0));
        if !self.verbose {
            model.silence();
        }
        match Self::solve(&vars, &mut model) {
            Ok((status, solution)) => {
                if self.verbose {
                    println!(
                        "IPOPT returned the following solution: {:?}",
                        solution.iter().map(Rational::to_f64).collect::<Vec<_>>()
                    );
                }
                if problem.holds_exact(&solution) {
                    println!("IPOPT found a solution that satisfies all constraints exactly.");
                    return Ok(solution);
                }
                match status {
                    SolutionStatus::Solved => {
                        println!(
                            "Solution by IPOPT does not actually satisfy the constraints (due to numerical issues)."
                        );
                        Err(SolverError::Failed)
                    }
                    SolutionStatus::Infeasible => {
                        println!("IPOPT found signs of infeasibility.");
                        Err(SolverError::MaybeInfeasible)
                    }
                    SolutionStatus::Other => {
                        println!("IPOPT failed to solve the problem.");
                        Err(SolverError::Other)
                    }
                }
            }
            Err(e) => {
                println!("IPOPT failed: {e}");
                Err(SolverError::Other)
            }
        }
    }
}

impl Optimizer for Ipopt {
    fn optimize(&mut self, problem: &ConstraintProblem, init: Vec<Rational>) -> Vec<Rational> {
        let init_obj = problem
            .objective
            .eval_exact(&init, &mut FxHashMap::default());
        let (vars, mut model) = self.construct_model(problem, Some(&init));
        if !self.verbose {
            model.silence();
        }
        match Self::solve(&vars, &mut model) {
            Ok((status, solution)) => {
                let cache = &mut FxHashMap::default();
                let obj_value = problem.objective.eval_exact(&solution, cache);
                let holds_exact = problem.holds_exact_with(&solution, cache);
                match status {
                    SolutionStatus::Solved => {
                        if holds_exact {
                            solution
                        } else {
                            println!("Solution by IPOPT does not actually satisfy the constraints (due to numerical issues).");
                            init
                        }
                    }
                    SolutionStatus::Infeasible => init,
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
                println!("IPOPT failed: {e}");
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
) -> (Expr, f64, f64) {
    let expr =
        constraint.lhs.to_ipopt_expr(vars, cache) - constraint.rhs.to_ipopt_expr(vars, cache);
    (expr, f64::NEG_INFINITY, -2.0 * tol)
}
