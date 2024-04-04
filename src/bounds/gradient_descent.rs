use std::time::Duration;

use good_lp::{Expression, ProblemVariables, Solution, SolverModel, VariableDefinition};
use ndarray::Array1;

use crate::bounds::sym_poly::PolyConstraint;

use super::{
    optimizer::Optimizer,
    solver::{ConstraintProblem, Solver, SolverError},
    sym_expr::{SymConstraint, SymExpr},
};

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
        let lr = self.step_size;
        let mut point: Array1<f64> = Array1::ones(problem.var_count);
        // initialize the polynomial coefficient values to their lower bounds (plus some slack)
        for constraint in &problem.constraints {
            if let Some(linear) = constraint.extract_linear() {
                let linear = linear.expr;
                let mut lower_bound = 1.0;
                let mut maybe_var = None;
                let mut num_vars = 0;
                for (var, coeff) in linear.coeffs.iter().enumerate() {
                    if coeff != &0.0 {
                        num_vars += 1;
                        if coeff > &0.0 {
                            lower_bound = -linear.constant / coeff;
                            maybe_var = Some(var);
                        }
                    }
                }
                if num_vars == 1 {
                    if let Some(var) = maybe_var {
                        if problem.coefficient_vars.contains(&var) {
                            point[var] = (lower_bound + 0.5) * 2.0; // TODO: is this robust enough?
                        }
                    }
                }
            }
        }
        let mut trajectory = vec![point.clone()];
        // Repeatedly update the point with the gradient of all the (almost) violated inequalities.
        // In most cases, one iteration should suffice.
        for _ in 0..self.max_iters {
            let violated_constraints = problem
                .constraints
                .iter()
                .filter(|c| !c.holds_exact(point.as_slice().unwrap()))
                .collect::<Vec<_>>();
            if violated_constraints.is_empty() {
                break;
            }
            let tight_constraints = problem
                .constraints
                .iter()
                .filter(|c| c.is_close(point.as_slice().unwrap(), self.step_size * self.step_size))
                .collect::<Vec<_>>();
            let gradients = tight_constraints
                .iter()
                .map(|c| Array1::from_vec(c.gradient_at(point.as_slice().unwrap())))
                .collect::<Vec<_>>();
            let direction = match self.dir_selection_strategy {
                DirSelectionStrategy::Greedy => largest_gradient(problem, &gradients),
                DirSelectionStrategy::ProjectPolyhedralCone => {
                    let mut direction = Array1::zeros(problem.var_count);
                    for gradient in &gradients {
                        direction += gradient;
                    }
                    project_into_cone(direction, &gradients, 1e-3)
                        .unwrap_or_else(|| largest_gradient(problem, &gradients))
                }
            };
            if let Some(update) = line_search(&point, &direction, &problem.constraints) {
                point += &update;
            } else {
                point += &(lr * direction);
            }
            trajectory.push(point.clone());
        }
        println!("Points:");
        for p in &trajectory {
            println!("{p},");
        }
        if !problem.holds_exact(point.as_slice().unwrap()) {
            return Err(SolverError::Timeout);
        }
        Ok(point.to_vec())
    }
}

fn line_search(
    point: &Array1<f64>,
    direction: &Array1<f64>,
    constraints: &[SymConstraint<f64>],
) -> Option<Array1<f64>> {
    let mut update = direction.clone();
    while !constraints
        .iter()
        .all(|c| c.holds_exact((point + &update).as_slice().unwrap()))
    {
        if update.iter().all(|x| x.abs() < 1e-9) {
            return None;
        }
        update *= 0.5;
    }
    loop {
        update *= 1.5;
        if !constraints
            .iter()
            .all(|c| c.holds_exact((point + &update).as_slice().unwrap()))
        {
            update /= 1.5;
            break;
        }
    }
    Some(update)
}

fn line_search_with_objective(
    point: &Array1<f64>,
    direction: &Array1<f64>,
    constraints: &[SymConstraint<f64>],
    objective: &SymExpr<f64>,
) -> Option<Array1<f64>> {
    let mut update = direction.clone();
    let mut best_obj = objective.eval(point.as_slice().unwrap());
    let mut best_update = None;
    loop {
        if update.iter().all(|x| x.abs() < 1e-9) {
            break;
        }
        let feasible = constraints
            .iter()
            .all(|c| c.holds_exact((point + &update).as_slice().unwrap()));
        let cur_obj = objective.eval((point + &update).as_slice().unwrap());
        if feasible && cur_obj < best_obj {
            best_obj = cur_obj;
            best_update = Some(update.clone());
        }
        update *= 0.5;
    }
    best_update.map(|mut best_update| {
        loop {
            let feasible = constraints
                .iter()
                .all(|c| c.holds_exact((point + &best_update).as_slice().unwrap()));
            let cur_obj = objective.eval((point + &best_update).as_slice().unwrap());
            if !feasible || cur_obj > best_obj {
                best_update /= 1.5;
                break;
            }
            best_update *= 1.5;
        }
        best_update
    })
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

fn project_into_cone(
    vector: Array1<f64>,
    good_dirs: &[Array1<f64>],
    epsilon: f64,
) -> Option<Array1<f64>> {
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

impl Optimizer for GradientDescent {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        objective: &SymExpr<f64>,
        init: Vec<f64>,
        _timeout: Duration,
    ) -> Vec<f64> {
        let slack_epsilon = 1e-6;
        let lr = self.step_size;
        let mut point: Array1<f64> = Array1::from_vec(init);
        let mut best_point: Array1<f64> = point.clone();
        let mut best_objective = objective.eval(point.as_slice().unwrap());
        let mut trajectory = vec![point.clone()];
        for _ in 0..500 {
            let tight_constraints = problem
                .constraints
                .iter()
                .filter(|c| c.is_close(point.as_slice().unwrap(), slack_epsilon))
                .collect::<Vec<_>>();
            let gradients = tight_constraints
                .iter()
                .map(|c| Array1::from_vec(c.gradient_at(point.as_slice().unwrap())))
                .collect::<Vec<_>>();
            let objective_grad =
                -Array1::from_vec(objective.gradient_at(point.as_slice().unwrap()));

            let direction = match self.dir_selection_strategy {
                DirSelectionStrategy::Greedy => largest_gradient(problem, &gradients),
                DirSelectionStrategy::ProjectPolyhedralCone => {
                    project_into_cone(objective_grad, &gradients, 1e-4)
                        .unwrap_or_else(|| largest_gradient(problem, &gradients))
                }
            };
            if let Some(update) =
                line_search_with_objective(&point, &direction, &problem.constraints, &objective)
            {
                point += &update;
            } else {
                point += &(lr * direction);
            }
            let objective = objective.eval(point.as_slice().unwrap());
            if objective < best_objective && problem.holds_exact(point.as_slice().unwrap()) {
                best_objective = objective;
                best_point = point.clone();
            }
            println!("Objective: {objective} at {point}");
            trajectory.push(point.clone());
        }
        println!("Points:");
        for p in &trajectory {
            println!("{p},");
        }
        println!("Best objective: {best_objective} at {best_point}");
        best_point.to_vec()
    }
}

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    dir_selection_strategy: DirSelectionStrategy,
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            dir_selection_strategy: DirSelectionStrategy::ProjectPolyhedralCone,
        }
    }
}

impl Optimizer for Adam {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        objective: &SymExpr<f64>,
        init: Vec<f64>,
        _timeout: Duration,
    ) -> Vec<f64> {
        let slack_epsilon = 1e-4;
        let epsilon = 1e-3;
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
        let mut point: Array1<f64> = Array1::from_vec(init);
        let mut best_point: Array1<f64> = point.clone();
        let mut best_objective = objective.eval(point.as_slice().unwrap());
        let mut trajectory = vec![point.clone()];
        let mut m = Array1::zeros(problem.var_count);
        let mut v = Array1::zeros(problem.var_count);
        let mut t = 1;
        for _ in 0..500 {
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
            let mut objective_grad =
                -Array1::from_vec(objective.gradient_at(point.as_slice().unwrap()));
            let obj_grad_len = objective_grad.dot(&objective_grad).sqrt();
            if obj_grad_len > 1e6 {
                objective_grad /= obj_grad_len;
                objective_grad *= 1e6;
            }
            gradients.push(objective_grad.clone());
            let mut direction = match self.dir_selection_strategy {
                DirSelectionStrategy::Greedy => largest_gradient(problem, &gradients),
                DirSelectionStrategy::ProjectPolyhedralCone => {
                    project_into_cone(objective_grad, &gradients, 1e-3)
                        .unwrap_or_else(|| largest_gradient(problem, &gradients))
                }
            };
            let dir_len = direction.dot(&direction).sqrt();
            if dir_len * epsilon > 0.1 {
                direction *= 0.1 / (dir_len * epsilon);
            }
            m = self.beta1 * &m + (1.0 - self.beta1) * &direction;
            v = self.beta2 * &v + (1.0 - self.beta2) * &(&direction * &direction);
            let m_hat = &m / (1.0 - self.beta1.powi(t));
            let v_hat = &v / (1.0 - self.beta2.powi(t));
            let update_dir = self.lr * &m_hat / (&v_hat.map(|x| x.sqrt()) + self.epsilon);
            let step_size = find_satisfying_in_dir(
                &point,
                &update_dir,
                &constraints_grads
                    .iter()
                    .map(|(c, _)| c.clone())
                    .collect::<Vec<_>>(),
            );
            point += &(step_size * update_dir);
            let objective = objective.eval(point.as_slice().unwrap());
            if objective < best_objective
                && constraints_grads
                    .iter()
                    .all(|(c, _)| c.holds(point.as_slice().unwrap()))
            {
                best_objective = objective;
                best_point = point.clone();
            }
            println!("Objective: {objective} at {point}");
            trajectory.push(point.clone());
            t += 1;
        }
        println!("Points:");
        for p in &trajectory {
            println!("{p},");
        }
        println!("Best objective: {best_objective} at {best_point}");
        best_point.to_vec()
    }
}

fn find_satisfying_in_dir(
    point: &Array1<f64>,
    dir: &Array1<f64>,
    constraints: &[PolyConstraint<f64>],
) -> f64 {
    let mut lo = 0.0;
    let mut hi = 1.0;
    while hi - lo > 1e-6 {
        let mid = (lo + hi) / 2.0;
        let new_point = point + &(mid * dir);
        if constraints
            .iter()
            .all(|c| c.holds(new_point.as_slice().unwrap()))
        {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}
