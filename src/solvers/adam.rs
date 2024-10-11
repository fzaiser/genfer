use std::time::Duration;

use ndarray::Array1;
use num_traits::Zero;
use rustc_hash::FxHashMap;

use crate::{
    numbers::Rational,
    sym_expr::SymExpr,
    util::{norm, normalize},
};

use super::{
    optimizer::Optimizer,
    solver::{ConstraintProblem, Solver, SolverError},
};

// Adam optimizer in comparison with the barrier method for constraints.
// The barrier is raised with each iteration of the optimization loop.
pub struct AdamBarrier {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    max_iter: usize,
    stopping_eps: f64,
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
            max_iter: 5000,
            stopping_eps: 1e-6,
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
        let mut problem = problem.clone();
        // Set objective to zero because we're just solving, not optimizing
        problem.objective = SymExpr::zero();
        let res = self.optimize(&problem, init, timeout);
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
        let mut best_point = Array1::from_vec(init).map(Rational::round);
        let mut point: Array1<f64> = best_point.clone();
        let mut best_objective =
            objective.eval_float(point.as_slice().unwrap(), &mut FxHashMap::default());
        let mut trajectory = vec![point.clone()];
        let mut m = Array1::zeros(problem.var_count);
        let mut v = Array1::zeros(problem.var_count);
        let mut t = 1;
        let constraints = problem.all_constraints().collect::<Vec<_>>();
        for _ in 0..self.max_iter {
            let cache = &mut FxHashMap::default();
            let grad_cache = &mut vec![FxHashMap::default(); point.len()];
            let obj = objective.eval_float(point.as_slice().unwrap(), &mut FxHashMap::default());
            let mut objective_grad =
                -Array1::from_vec(objective.gradient_at(point.as_slice().unwrap(), grad_cache))
                    / obj;
            // let mut barrier_penalty = 0.0;
            for c in &constraints {
                let constraint_grad =
                    Array1::from_vec(c.gradient_at(point.as_slice().unwrap(), grad_cache));
                let dist = c.estimate_signed_dist(point.as_slice().unwrap(), cache, grad_cache);
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
            let objective =
                objective.eval_float(point.as_slice().unwrap(), &mut FxHashMap::default());
            if objective <= best_objective && problem.holds_exact_f64(point.as_slice().unwrap()) {
                best_objective = objective;
                best_point = point.clone();
            }
            trajectory.push(point.clone());
            if norm(&update_dir.view()) < self.stopping_eps {
                break;
            }
            t += 1;
        }
        if self.verbose {
            println!("Points:");
            for p in trajectory.iter().step_by(t as usize / 100) {
                println!("{p},");
            }
        }
        println!("Best objective: {best_objective} at {best_point}");
        best_point.mapv(Rational::from).to_vec()
    }
}
