use std::ops::{Add, Div, Mul, Neg, Sub};

use good_lp::{
    solvers::coin_cbc::CoinCbcProblem, variable, ProblemVariables, Solution, SolverModel, Variable,
};
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

use crate::{
    numbers::{FloatNumber, Rational},
    sym_expr::SymExpr,
};

use super::{problem::ConstraintProblem, Optimizer};

pub struct LinearProgrammingOptimizer;

impl Optimizer for LinearProgrammingOptimizer {
    fn optimize(&mut self, problem: &ConstraintProblem, init: Vec<Rational>) -> Vec<Rational> {
        if let Some(solution) = optimize_linear_parts(problem, init.clone()) {
            let init_obj = problem
                .objective
                .eval_exact(&init, &mut FxHashMap::default());
            let objective_value = problem
                .objective
                .eval_exact(&solution, &mut FxHashMap::default());
            if init_obj <= objective_value {
                println!("LP solver failed to improve the objective.");
                return init;
            }
            solution
        } else {
            println!("LP solver failed; returning previous solution.");
            init
        }
    }
}

fn extract_lp(problem: &ConstraintProblem, init: &[Rational]) -> LinearProblem {
    let mut replacements = (0..problem.var_count).map(SymExpr::var).collect::<Vec<_>>();
    for v in problem.decay_vars.iter().chain(problem.factor_vars.iter()) {
        replacements[*v] = SymExpr::from(init[*v].clone());
    }
    let problem = problem.substitute(&replacements);
    let cache = &mut FxHashMap::default();
    let constraints = problem
        .constraints
        .iter()
        .map(|constraint| constraint.extract_linear(cache).unwrap())
        .collect::<Vec<_>>();
    let objective = problem.objective.extract_linear(cache).unwrap();
    let mut var_bounds = problem.var_bounds.clone();
    for v in problem.decay_vars.iter().chain(problem.factor_vars.iter()) {
        var_bounds[*v] = (init[*v].clone(), init[*v].clone());
    }
    LinearProblem {
        objective,
        constraints,
        var_bounds,
    }
}

fn create_cbc_model(problem: &LinearProblem, tighten: f64) -> (Vec<Variable>, CoinCbcProblem) {
    let mut lp = ProblemVariables::new();
    let var_list = problem
        .var_bounds
        .iter()
        .map(|(lo, hi)| {
            let var = variable().min(lo.to_f64());
            let var = if hi.is_finite() {
                var.max(hi.to_f64())
            } else {
                var
            };
            lp.add(var)
        })
        .collect::<Vec<_>>();
    let objective = problem
        .objective
        .normalize()
        .to_lp_expr(&var_list, &Rational::to_f64_up);
    let mut lp = lp.minimise(objective).using(good_lp::default_solver);
    for constraint in &problem.constraints {
        lp.add_constraint(
            constraint
                .normalize()
                .tighten(tighten)
                .to_lp_constraint(&var_list),
        );
    }
    (var_list, lp)
}

pub fn optimize_linear_parts(
    problem: &ConstraintProblem,
    init: Vec<Rational>,
) -> Option<Vec<Rational>> {
    println!("Running LP solver CBC...");
    let lp = extract_lp(problem, &init);
    let mut tol = 1e-9;
    let mut tighten = 1e-9;
    let retries = 8;
    for retry in 0..retries {
        if retry > 0 {
            println!("Retrying with modified numerical tolerances...");
        }
        let (vars, mut model) = create_cbc_model(&lp, tighten);
        // Tolerance during pre-solve phase:
        model.set_parameter("preT", &tol.to_string());
        // For a feasible solution no primal infeasibility, i.e., constraint violation, may exceed this value:
        model.set_parameter("primalT", &tol.to_string());
        // For an optimal solution no dual infeasibility may exceed this value:
        model.set_parameter("dualT", &tol.to_string());
        // The log level adjustments don't seem to have an effect, but we set them anyway:
        model.set_parameter("slogLevel", "0");
        model.set_parameter("logLevel", "0");
        model.set_parameter("messages", "off");
        let solution = match model.solve() {
            Ok(solution) => solution,
            Err(good_lp::ResolutionError::Unbounded) => {
                println!("LP solver found the problem unbounded.");
                return None;
            }
            Err(good_lp::ResolutionError::Infeasible) => {
                println!("LP solver found the problem infeasible.");
                tol *= 1.5;
                tighten /= 2.0;
                continue;
            }
            Err(good_lp::ResolutionError::Other(msg)) => {
                todo!("Other error: {msg}");
            }
            Err(good_lp::ResolutionError::Str(msg)) => {
                todo!("Error: {msg}");
            }
        };
        let solution = vars
            .iter()
            .map(|v| Rational::from(solution.value(*v)))
            .collect::<Vec<_>>();
        if problem.holds_exact(&solution) {
            if retry > 0 {
                println!("LP solver succeeded after {retry} retries (due to numerical issues).");
            }
            return Some(solution);
        }
        println!("Solution by LP solver does not satisfy the constraints (rounding issues).");
        tol *= 2.0;
        tighten *= 3.0;
    }
    println!("LP solver failed after {retries} retries (probably numerical issues).");
    None
}

pub struct LinearProblem {
    pub objective: LinearExpr,
    pub constraints: Vec<LinearConstraint>,
    pub var_bounds: Vec<(Rational, Rational)>,
}

#[derive(Clone, Debug)]
pub struct LinearExpr {
    pub coeffs: Vec<Rational>,
    pub constant: Rational,
}

impl LinearExpr {
    pub(crate) fn new(coeffs: Vec<Rational>, constant: Rational) -> Self {
        Self { coeffs, constant }
    }

    pub fn zero() -> Self {
        Self::new(vec![], Rational::zero())
    }

    pub(crate) fn one() -> Self {
        Self::new(vec![Rational::one()], Rational::zero())
    }

    pub(crate) fn constant(constant: Rational) -> Self {
        Self::new(vec![], constant)
    }

    pub(crate) fn var(var: usize) -> Self {
        let mut coeffs = vec![Rational::zero(); var + 1];
        coeffs[var] = Rational::one();
        Self::new(coeffs, Rational::zero())
    }

    pub(crate) fn as_constant(&self) -> Option<&Rational> {
        if self.coeffs.iter().all(Rational::is_zero) {
            Some(&self.constant)
        } else {
            None
        }
    }

    pub(crate) fn to_lp_expr(
        &self,
        vars: &[good_lp::Variable],
        conv: &impl Fn(&Rational) -> f64,
    ) -> good_lp::Expression {
        let mut result = good_lp::Expression::from(conv(&self.constant));
        for (coeff, var) in self.coeffs.iter().zip(vars) {
            result.add_mul(conv(coeff), var);
        }
        result
    }

    pub(crate) fn max_coeff(&self) -> Rational {
        self.coeffs.iter().fold(self.constant.clone(), |max, c| {
            if c > &max {
                c.clone()
            } else {
                max
            }
        })
    }

    pub(crate) fn normalize(&self) -> Self {
        let max = self.max_coeff();
        let scale = if max.is_zero() {
            Rational::one()
        } else {
            Rational::one() / max
        };
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| c.clone() * scale.clone())
            .collect();
        let constant = self.constant.clone() * scale;
        Self::new(coeffs, constant)
    }
}

impl std::fmt::Display for LinearExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            if coeff.is_one() {
                write!(f, "{}", SymExpr::var(i))?;
            } else if *coeff == -Rational::one() {
                write!(f, "-{}", SymExpr::var(i))?;
            } else {
                write!(f, "{}*{}", coeff, SymExpr::var(i))?;
            }
        }
        if !self.constant.is_zero() {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.constant)?;
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl Neg for LinearExpr {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.constant = -self.constant;
        for coeff in &mut self.coeffs {
            *coeff = -coeff.clone();
        }
        self
    }
}

impl Add for LinearExpr {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let mut constant = self.constant;
        constant += other.constant;
        let (mut coeffs, other) = if self.coeffs.len() > other.coeffs.len() {
            (self.coeffs, other.coeffs)
        } else {
            (other.coeffs, self.coeffs)
        };
        for (c1, c2) in coeffs.iter_mut().zip(other) {
            *c1 += c2;
        }
        Self::new(coeffs, constant)
    }
}

impl Sub for LinearExpr {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Mul<Rational> for LinearExpr {
    type Output = Self;

    #[inline]
    fn mul(self, other: Rational) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c * other.clone()).collect(),
            self.constant * other.clone(),
        )
    }
}

impl Div<Rational> for LinearExpr {
    type Output = Self;

    #[inline]
    fn div(self, other: Rational) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c / other.clone()).collect(),
            self.constant / other.clone(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct LinearConstraint {
    pub expr: LinearExpr,
    /// If true, `expr` must be equal to zero, otherwise it must be nonnegative
    pub eq_zero: bool,
}

impl LinearConstraint {
    pub(crate) fn le(e1: LinearExpr, e2: LinearExpr) -> Self
    where
        LinearExpr: Sub<Output = LinearExpr>,
    {
        Self {
            expr: e2 - e1,
            eq_zero: false,
        }
    }

    pub(crate) fn to_lp_constraint(&self, var_list: &[good_lp::Variable]) -> good_lp::Constraint {
        let result = self.expr.to_lp_expr(var_list, &Rational::to_f64_down);
        if self.eq_zero {
            result.eq(0.0)
        } else {
            result.geq(0.0)
        }
    }

    pub(crate) fn normalize(&self) -> Self {
        Self {
            expr: self.expr.normalize(),
            eq_zero: self.eq_zero,
        }
    }

    pub(crate) fn tighten(&self, eps: f64) -> Self {
        let constant = if self.expr.max_coeff().is_zero() {
            self.expr.constant.clone()
        } else {
            self.expr.constant.clone() - Rational::from(eps)
        };
        Self {
            expr: LinearExpr::new(self.expr.coeffs.clone(), constant),
            eq_zero: self.eq_zero,
        }
    }
}

impl std::fmt::Display for LinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.eq_zero {
            write!(f, "{} = 0", self.expr)
        } else {
            write!(f, "{} >= 0", self.expr)
        }
    }
}
