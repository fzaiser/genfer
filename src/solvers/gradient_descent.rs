use std::time::Duration;

use good_lp::{Expression, ProblemVariables, Solution, SolverModel, VariableDefinition};
use ndarray::Array1;

use crate::{
    numbers::Rational,
    sym_expr::{SymConstraint, SymExpr},
    util::normalize,
};

use super::{
    optimizer::Optimizer,
    solver::{ConstraintProblem, Solver, SolverError},
};

pub enum DirSelectionStrategy {
    Greedy,
    ProjectPolyhedralCone,
}

pub struct GradientDescent {
    pub step_size: f64,
    pub max_iters: usize,
    pub dir_selection_strategy: DirSelectionStrategy,
    verbose: bool,
}

impl GradientDescent {
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self {
            step_size: 1e-4,
            max_iters: 100,
            dir_selection_strategy: DirSelectionStrategy::ProjectPolyhedralCone,
            verbose: false,
        }
    }
}

impl Solver for GradientDescent {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        _timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError> {
        let lr = self.step_size;
        let init = vec![1.0 - 1e-9; problem.var_count];
        let mut point: Array1<f64> = Array1::from_vec(init);
        // initialize the polynomial coefficient values to their lower bounds (plus some slack)
        for (v, (lo, _)) in problem.var_bounds.iter().enumerate() {
            if problem.block_vars.contains(&v) {
                point[v] = (lo.round_to_f64() + 0.5) * 2.0; // TODO: is this robust enough?
            }
        }
        let mut trajectory = vec![point.clone()];
        let constraints = problem.all_constraints().collect::<Vec<_>>();
        // Repeatedly update the point with the gradient of all the (almost) violated inequalities.
        // In most cases, one iteration should suffice.
        for _ in 0..self.max_iters {
            let violated_constraints = constraints
                .iter()
                .filter(|c| !c.holds_exact(point.map(|f| Rational::from(*f)).as_slice().unwrap()))
                .collect::<Vec<_>>();
            if violated_constraints.is_empty() {
                break;
            }
            let tight_constraints = constraints
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
            if let Some(update) = line_search(&point, &direction, &constraints) {
                point += &update;
            } else {
                point += &(lr * direction);
            }
            trajectory.push(point.clone());
        }
        if self.verbose {
            println!("Points:");
            for p in &trajectory {
                println!("{p},");
            }
        }
        let exact_point = point.mapv(Rational::from);
        if !problem.holds_exact(exact_point.as_slice().unwrap()) {
            return Err(SolverError::Timeout);
        }
        Ok(exact_point.to_vec())
    }
}

fn line_search(
    point: &Array1<f64>,
    direction: &Array1<f64>,
    constraints: &[SymConstraint],
) -> Option<Array1<f64>> {
    let mut update = direction.clone();
    while !constraints
        .iter()
        .all(|c| c.holds_exact_f64((point + &update).as_slice().unwrap()))
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
            .all(|c| c.holds_exact_f64((point + &update).as_slice().unwrap()))
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
    constraints: &[SymConstraint],
    objective: &SymExpr,
) -> Option<Array1<f64>> {
    let mut update = direction.clone();
    let mut best_obj = objective.eval_float(point.as_slice().unwrap());
    let mut best_update = None;
    loop {
        if update.iter().all(|x| x.abs() < 1e-9) {
            break;
        }
        let feasible = constraints
            .iter()
            .all(|c| c.holds_exact_f64((point + &update).as_slice().unwrap()));
        let cur_obj = objective.eval_float((point + &update).as_slice().unwrap());
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
                .all(|c| c.holds_exact_f64((point + &best_update).as_slice().unwrap()));
            let cur_obj = objective.eval_float((point + &best_update).as_slice().unwrap());
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
        init: Vec<Rational>,
        _timeout: Duration,
    ) -> Vec<Rational> {
        let slack_epsilon = 1e-6;
        let objective = &problem.objective;
        let lr = self.step_size;
        let mut point: Array1<f64> = Array1::from_vec(init).map(Rational::round_to_f64);
        let mut best_point: Array1<f64> = point.clone();
        let mut best_objective = objective.eval_float(point.as_slice().unwrap());
        let mut trajectory = vec![point.clone()];
        let constraints = problem.all_constraints().collect::<Vec<_>>();
        for _ in 0..500 {
            let tight_constraints = constraints
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
                line_search_with_objective(&point, &direction, &constraints, objective)
            {
                point += &update;
            } else {
                point += &(lr * direction);
            }
            let objective = objective.eval_float(point.as_slice().unwrap());
            if objective < best_objective && problem.holds_exact_f64(point.as_slice().unwrap()) {
                best_objective = objective;
                best_point = point.clone();
            }
            if self.verbose {
                println!("Objective: {objective} at {point}");
            }
            trajectory.push(point.clone());
        }
        if self.verbose {
            println!("Points:");
            for p in &trajectory {
                println!("{p},");
            }
        }
        println!("Best objective: {best_objective} at {best_point}");
        best_point.mapv(Rational::from).to_vec()
    }
}

pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    dir_selection_strategy: DirSelectionStrategy,
    verbose: bool,
}

impl Adam {
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            dir_selection_strategy: DirSelectionStrategy::ProjectPolyhedralCone,
            verbose: false,
        }
    }
}

impl Optimizer for Adam {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        init: Vec<Rational>,
        _timeout: Duration,
    ) -> Vec<Rational> {
        let slack_epsilon = 1e-6;
        let objective = &problem.objective;
        let mut best_point = Array1::from_vec(init);
        let mut point = best_point.map(Rational::round_to_f64);
        let mut best_objective = objective.eval_float(point.as_slice().unwrap());
        let mut trajectory = vec![point.clone()];
        let mut m = Array1::zeros(problem.var_count);
        let mut v = Array1::zeros(problem.var_count);
        let mut t = 1;
        let constraints = problem.all_constraints().collect::<Vec<_>>();
        for _ in 0..500 {
            let tight_constraints = constraints
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
            m = self.beta1 * &m + (1.0 - self.beta1) * &direction;
            v = self.beta2 * &v + (1.0 - self.beta2) * &(&direction * &direction);
            let m_hat = &m / (1.0 - self.beta1.powi(t));
            let v_hat = &v / (1.0 - self.beta2.powi(t));
            let update_dir = self.lr * &m_hat / (&v_hat.map(|x| x.sqrt()) + self.epsilon);
            if let Some(update) =
                line_search_with_objective(&point, &direction, &constraints, objective)
            {
                point += &update;
            } else {
                point += &update_dir;
            };
            let objective = objective.eval_float(point.as_slice().unwrap());
            let point_exact = point.mapv(Rational::from);
            if objective < best_objective && problem.holds_exact(point_exact.as_slice().unwrap()) {
                best_objective = objective;
                best_point = point_exact;
            }
            if self.verbose {
                println!("Objective: {objective} at {point}");
            }
            trajectory.push(point.clone());
            t += 1;
        }
        if self.verbose {
            println!("Points:");
            for p in &trajectory {
                println!("{p},");
            }
        }
        println!(
            "Best objective: {best_objective} at {}",
            best_point.mapv(|r| r.round_to_f64())
        );
        best_point.to_vec()
    }
}

// Adam optimizer in comparison with the barrier method for constraints.
// The barrier is raised with each iteration of the optimization loop.
pub struct AdamBarrier {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    verbose: bool,
}

impl AdamBarrier {
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Default for AdamBarrier {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            verbose: false,
        }
    }
}

impl Solver for AdamBarrier {
    fn solve(
        &mut self,
        problem: &ConstraintProblem,
        timeout: Duration,
    ) -> Result<Vec<Rational>, SolverError> {
        let init = vec![Rational::from(1.0 - 1e-3); problem.var_count];
        let res = self.optimize(problem, init, timeout);
        if problem.holds_exact(&res) {
            Ok(res)
        } else {
            Err(SolverError::Timeout)
        }
    }
}

impl Optimizer for AdamBarrier {
    fn optimize(
        &mut self,
        problem: &ConstraintProblem,
        init: Vec<Rational>,
        _timeout: Duration,
    ) -> Vec<Rational> {
        let objective = &problem.objective;
        let mut best_point = Array1::from_vec(init);
        let mut point: Array1<f64> = best_point.map(Rational::round_to_f64);
        let mut best_objective = objective.eval_float(point.as_slice().unwrap());
        let mut trajectory = vec![point.clone()];
        let mut m = Array1::zeros(problem.var_count);
        let mut v = Array1::zeros(problem.var_count);
        let mut t = 1;
        let constraints = problem.all_constraints().collect::<Vec<_>>();
        for _ in 0..5000 {
            let obj = objective.eval_float(point.as_slice().unwrap());
            let mut objective_grad =
                -Array1::from_vec(objective.gradient_at(point.as_slice().unwrap())) / obj;
            // let mut barrier_penalty = 0.0;
            for c in &constraints {
                let constraint_grad = Array1::from_vec(c.gradient_at(point.as_slice().unwrap()));
                let dist = c.estimate_signed_dist(point.as_slice().unwrap());
                let concentration = f64::from(t);
                let barrier_grad = (concentration * dist).exp()
                    * concentration
                    * normalize(&constraint_grad.view());
                objective_grad += &barrier_grad;
                // barrier_penalty += (concentration * dist).exp();
            }
            let direction = objective_grad;
            m = self.beta1 * &m + (1.0 - self.beta1) * &direction;
            v = self.beta2 * &v + (1.0 - self.beta2) * &(&direction * &direction);
            let m_hat = &m / (1.0 - self.beta1.powi(t));
            let v_hat = &v / (1.0 - self.beta2.powi(t));
            let update_dir = self.lr * &m_hat / (&v_hat.map(|x| x.sqrt()) + self.epsilon);
            point += &update_dir;
            let objective = objective.eval_float(point.as_slice().unwrap());
            let point_exact = point.mapv(Rational::from);
            if objective <= best_objective && problem.holds_exact(point_exact.as_slice().unwrap()) {
                best_objective = objective;
                best_point = point_exact;
            }
            trajectory.push(point.clone());
            t += 1;
        }
        if self.verbose {
            println!("Points:");
            for p in trajectory.iter().step_by(t as usize / 100) {
                println!("{p},");
            }
        }
        println!(
            "Best objective: {best_objective} at {}",
            best_point.mapv(|r| r.round_to_f64())
        );
        best_point.to_vec()
    }
}
