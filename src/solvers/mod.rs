pub mod adam;
pub mod ipopt;
pub mod linear;
pub mod problem;
pub mod z3;

use std::time::Duration;

use crate::numbers::Rational;

use problem::ConstraintProblem;

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
    Other,
}

pub trait Solver {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError>;
}

pub trait Optimizer {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        init: Vec<Rational>,
        timeout: Duration,
    ) -> Vec<Rational>;
}
