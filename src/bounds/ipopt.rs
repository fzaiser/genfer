use std::time::Duration;

use crate::number::Rational;

use super::{
    optimizer::Optimizer,
    solver::{ConstraintProblem, Solver, SolverError},
    sym_expr::{SymConstraint, SymExpr, SymExprKind},
};

use descent::{
    expr::{dynam::Expr, Var},
    model::{Model, SolutionStatus},
};
use descent_ipopt::IpoptModel;
use num_traits::Zero;

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
        let mut model = IpoptModel::new();
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
        for constraint in problem.all_constraints() {
            let (expr, lo, hi) = ipopt_constraint(&constraint, &vars, self.tol);
            model.add_con(expr, lo, hi);
        }
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
            tol: 1e-8,
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
        model.set_obj(ipopt_expr(&SymExpr::zero(), &vars));
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
        objective: &SymExpr,
        init: Vec<Rational>,
        _timeout: Duration,
    ) -> Vec<Rational> {
        let (vars, mut model) = self.construct_model(problem, Some(&init));
        model.set_obj(ipopt_expr(objective, &vars));
        match Self::solve(&vars, &mut model) {
            Ok((status, solution)) => match status {
                SolutionStatus::Solved => {
                    println!("IPOPT found the following solution:");
                    print!(
                        "Objective: {} at {:?}",
                        objective.eval_exact(&solution).round_to_f64(),
                        solution
                            .iter()
                            .map(Rational::round_to_f64)
                            .collect::<Vec<_>>()
                    );
                    if problem.holds_exact(&solution) {
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
                    println!("IPOPT failed to solve the problem; returning initial solution.");
                    init
                }
            },
            Err(e) => {
                println!("IPOPT failed: {e}\nReturning initial solution.");
                init
            }
        }
    }
}

fn ipopt_expr(expr: &SymExpr, vars: &[Var]) -> Expr {
    match expr.kind() {
        SymExprKind::Constant(c) => c.float().into(),
        SymExprKind::Variable(v) => vars[*v].into(),
        SymExprKind::Add(lhs, rhs) => ipopt_expr(lhs, vars) + ipopt_expr(rhs, vars),
        SymExprKind::Mul(lhs, rhs) => ipopt_expr(lhs, vars) * ipopt_expr(rhs, vars),
        SymExprKind::Pow(base, exp) => {
            use descent::expr::dynam::NumOps;
            ipopt_expr(base, vars).powi(*exp)
        }
    }
}

fn ipopt_constraint(constraint: &SymConstraint, vars: &[Var], tol: f64) -> (Expr, f64, f64) {
    match constraint {
        SymConstraint::Eq(lhs, rhs) => {
            let expr = ipopt_expr(lhs, vars) - ipopt_expr(rhs, vars);
            (expr, 0.0, 0.0)
        }
        SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
            let expr = ipopt_expr(lhs, vars) - ipopt_expr(rhs, vars);
            (expr, f64::NEG_INFINITY, -tol)
        }
        SymConstraint::Or(_) => (0.0.into(), 0.0, 0.0),
    }
}
