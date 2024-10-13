use std::time::Duration;

use super::{problem::ConstraintProblem, Solver, SolverError};
use crate::{
    numbers::{FloatNumber, Rational},
    util::{rational_to_z3, z3_real_to_rational},
};

pub struct Z3Solver;

impl Solver for Z3Solver {
    fn solve(&mut self, problem: &ConstraintProblem) -> Result<Vec<Rational>, SolverError> {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        let timeout = Duration::from_secs(10);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for (v, (lo, hi)) in problem.var_bounds.iter().enumerate() {
            let var = z3::ast::Real::new_const(&ctx, v as u32);
            let lo = rational_to_z3(&ctx, lo);
            solver.assert(&lo.le(&var));
            if hi.is_finite() {
                let hi = rational_to_z3(&ctx, hi);
                solver.assert(&var.lt(&hi));
            }
        }
        for constraint in &problem.constraints {
            solver.assert(&constraint.to_z3(&ctx, &rational_to_z3));
        }
        solver.push();
        match solver.check() {
            z3::SatResult::Unknown => {
                if let Some(reason) = solver.get_reason_unknown() {
                    if reason == "timeout" {
                        return Err(SolverError::Failed);
                    }
                    panic!("Solver responded 'unknown': {reason}")
                } else {
                    panic!("Solver responded 'unknown' but no reason was given.")
                }
            }
            z3::SatResult::Unsat => return Err(SolverError::ProvenInfeasible),
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
                assignment.push(val);
            }
            assignment
        } else {
            panic!("Solver returned SAT but no model.")
        };
        Ok(assignment)
    }
}
