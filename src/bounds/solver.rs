use std::time::Duration;

use crate::{
    bounds::util::{rational_to_z3, z3_real_to_rational},
    number::Rational,
};

use super::sym_expr::{SymConstraint, SymExpr};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
}

pub struct ConstraintProblem {
    pub var_count: usize,
    pub geom_vars: Vec<usize>,
    pub factor_vars: Vec<usize>,
    pub coefficient_vars: Vec<usize>,
    pub constraints: Vec<SymConstraint>,
}

impl ConstraintProblem {
    pub fn holds_exact_f64(&self, assignment: &[f64]) -> bool {
        let assignment = assignment
            .iter()
            .map(|r| Rational::from(*r))
            .collect::<Vec<_>>();
        self.holds_exact(&assignment)
    }

    pub fn holds_exact(&self, assignment: &[Rational]) -> bool {
        self.constraints.iter().all(|c| c.holds_exact(assignment))
    }
}

pub trait Solver {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError>;
}

pub struct Z3Solver;

impl Solver for Z3Solver {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError> {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for constraint in &problem.constraints {
            solver.assert(&constraint.to_z3(&ctx, &rational_to_z3));
        }
        solver.push();
        match solver.check() {
            z3::SatResult::Unknown => {
                if let Some(reason) = solver.get_reason_unknown() {
                    if reason == "timeout" {
                        return Err(SolverError::Timeout);
                    }
                    panic!("Solver responded 'unknown': {reason}")
                } else {
                    panic!("Solver responded 'unknown' but no reason was given.")
                }
            }
            z3::SatResult::Unsat => return Err(SolverError::Infeasible),
            z3::SatResult::Sat => {}
        }
        let assignment = if let Some(model) = solver.get_model() {
            let mut assignment = Vec::new();
            for var in 0..problem.var_count {
                let val = model
                    .eval(&z3::ast::Real::new_const(&ctx, var as u32), false)
                    .unwrap();
                let val = z3_real_to_rational(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to rational"));
                println!("{var} -> {val}", var = SymExpr::var(var));
                assignment.push(val);
            }
            assignment
        } else {
            panic!("Solver returned SAT but no model.")
        };
        Ok(assignment)
    }
}
