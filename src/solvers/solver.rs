use std::time::Duration;

use crate::{
    numbers::{FloatNumber, Rational},
    sym_expr::{SymConstraint, SymExpr},
    util::{rational_to_z3, z3_real_to_rational},
};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
    Other,
}

pub struct ConstraintProblem {
    pub var_count: usize,
    pub decay_vars: Vec<usize>,
    pub factor_vars: Vec<usize>,
    pub block_vars: Vec<usize>,
    pub var_bounds: Vec<(Rational, Rational)>,
    pub constraints: Vec<SymConstraint>,
    /// Optimization objective (zero means no optimization)
    pub objective: SymExpr,
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
        self.all_constraints().all(|c| c.holds_exact(assignment))
    }

    pub fn all_constraints(&self) -> impl Iterator<Item = SymConstraint> + '_ {
        self.var_bounds
            .iter()
            .enumerate()
            .flat_map(move |(var, (lo, hi))| {
                let first = SymExpr::var(var).must_ge(lo.clone().into());
                if hi.is_finite() {
                    let second = SymExpr::var(var).must_lt(hi.clone().into());
                    vec![first, second]
                } else {
                    vec![first]
                }
            })
            .chain(self.constraints.iter().cloned())
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
        for constraint in problem.all_constraints() {
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
                assignment.push(val);
            }
            assignment
        } else {
            panic!("Solver returned SAT but no model.")
        };
        Ok(assignment)
    }
}
