use std::time::Duration;

use good_lp::{Expression, ProblemVariables, Solution, SolverModel, VariableDefinition};
use ndarray::Array1;

use crate::bounds::sym_poly::PolyConstraint;

use super::solver::{ConstraintProblem, Solver, SolverError};

pub enum DirSelectionStrategy {
    Greedy,
    ProjectPolyhedralCone,
}

pub struct GradientDescent {
    pub step_size: f64,
    pub max_iters: usize,
    pub dir_selection_strategy: DirSelectionStrategy,
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            step_size: 1e-4,
            max_iters: 100,
            dir_selection_strategy: DirSelectionStrategy::ProjectPolyhedralCone,
        }
    }
}

impl Solver for GradientDescent {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        _timeout: Duration,
    ) -> Result<Vec<f64>, SolverError> {
        let slack_epsilon = self.step_size * self.step_size;
        let epsilon = self.step_size;
        let all_vars = (0..problem.var_count).collect::<Vec<_>>();
        let constraints_grads = problem
            .constraints
            .iter()
            .filter_map(|c| {
                let c = c.to_poly();
                let grad = match &c {
                    PolyConstraint::Le(lhs, rhs) | PolyConstraint::Lt(lhs, rhs) => {
                        Array1::from_vec((rhs.clone() - lhs.clone()).gradient(&all_vars))
                    }
                    _ => return None,
                };
                Some((c, grad))
            })
            .collect::<Vec<_>>();
        let mut point: Array1<f64> = Array1::ones(problem.var_count);
        let mut trajectory = vec![point.clone()];
        for _ in 0..self.max_iters {
            let mut gradients = Vec::new();
            for (constraint, gradient) in &constraints_grads {
                if constraint.has_slack(point.as_slice().unwrap(), slack_epsilon) {
                    continue;
                }
                let gradient = gradient.map(|g| g.eval(point.as_slice().unwrap()));
                if gradient.iter().all(|x| x == &0.0) {
                    continue;
                }
                gradients.push(gradient);
            }
            if gradients.is_empty() {
                break;
            }
            let direction = match self.dir_selection_strategy {
                DirSelectionStrategy::Greedy => largest_gradient(problem, &gradients),
                DirSelectionStrategy::ProjectPolyhedralCone => {
                    let mut direction = Array1::zeros(problem.var_count);
                    for gradient in &gradients {
                        direction += gradient;
                    }
                    project_into_cone(direction, &gradients)
                        .unwrap_or_else(|| largest_gradient(problem, &gradients))
                }
            };
            point += &(epsilon * direction);
            trajectory.push(point.clone());
        }
        println!("Points:");
        for p in &trajectory {
            println!("{p},");
        }
        if constraints_grads
            .iter()
            .any(|(c, _)| !c.holds(point.as_slice().unwrap()))
        {
            return Err(SolverError::Timeout);
        }
        Ok(point.to_vec())
    }
}

fn largest_gradient(problem: &ConstraintProblem, gradients: &[Array1<f64>]) -> Array1<f64> {
    let mut best_gradient = Array1::zeros(problem.var_count);
    for gradient in gradients {
        if gradient.dot(gradient) > best_gradient.dot(&best_gradient) {
            best_gradient = gradient.clone();
        }
    }
    best_gradient
}

fn project_into_cone(vector: Array1<f64>, good_dirs: &[Array1<f64>]) -> Option<Array1<f64>> {
    let epsilon = 1e-3;
    if good_dirs.iter().all(|dir| dir.dot(&vector) >= 0.0) {
        return Some(vector);
    }
    let mut lp = ProblemVariables::new();
    let vars = lp.add_vector(VariableDefinition::new().min(0.0).max(1.0), good_dirs.len());
    let mut objective = Expression::from(0.0);
    for (dir, var) in good_dirs.iter().zip(&vars) {
        objective.add_mul(dir.dot(&vector), var);
    }
    let mut problem = lp.maximise(objective).using(good_lp::default_solver);
    for cur_dir in good_dirs {
        let mut lhs = Expression::from(0.0);
        for (dir, var) in good_dirs.iter().zip(&vars) {
            lhs.add_mul(dir.dot(cur_dir), var);
        }
        problem.add_constraint(lhs.geq(epsilon * cur_dir.dot(cur_dir)));
    }
    let solution = match problem.solve() {
        Ok(solution) => solution,
        Err(err) => {
            match err {
                good_lp::ResolutionError::Unbounded => unreachable!(),
                good_lp::ResolutionError::Infeasible => {
                    println!("LP solver found the problem infeasible.");
                }
                good_lp::ResolutionError::Other(msg) => println!("Other error: {msg}"),
                good_lp::ResolutionError::Str(msg) => println!("Error: {msg}"),
            }
            return None;
        }
    };
    let solution = vars.iter().map(|v| solution.value(*v)).collect::<Vec<_>>();
    let mut result = Array1::zeros(vector.len());
    for (dir, var) in good_dirs.iter().zip(solution) {
        result += &(var * dir);
    }
    Some(result)
}
