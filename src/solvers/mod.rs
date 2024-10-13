pub mod adam;
pub mod ipopt;
pub mod linear;
pub mod problem;
pub mod z3;

use crate::numbers::Rational;

use problem::ConstraintProblem;

#[derive(Clone, Debug)]
pub enum SolverError {
    ProvenInfeasible,
    MaybeInfeasible,
    Failed,
    Other,
}

pub trait Solver {
    fn solve(&mut self, problem: &ConstraintProblem) -> Result<Vec<Rational>, SolverError>;
}

pub trait Optimizer {
    fn optimize(&mut self, problem: &ConstraintProblem, init: Vec<Rational>) -> Vec<Rational>;
}
