use crate::{
    bounds::{sym_expr::SymExpr, sym_poly::SymPolynomial},
    multivariate_taylor::TaylorPoly,
    number::Number,
    ppl::Var,
    semantics::support::VarSupport,
    support::SupportSet,
};
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct BoundResult {
    pub bound: GeometricBound,
    pub var_supports: VarSupport,
}

impl BoundResult {
    pub fn marginalize(self, var: Var) -> BoundResult {
        let mut var_supports = self.var_supports;
        if !var_supports[var].is_empty() {
            var_supports.set(var, SupportSet::zero());
        }
        BoundResult {
            bound: self.bound.marginalize(var),
            var_supports,
        }
    }
}

impl std::fmt::Display for BoundResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(support_vec) = self.var_supports.as_vec() {
            for (i, support) in support_vec.iter().enumerate() {
                writeln!(
                    f,
                    "Support of {var}: {support}",
                    var = Var(i),
                    support = support
                )?;
            }
        } else {
            writeln!(f, "Support: empty")?;
        }
        writeln!(f, "Bound:\n{bound}", bound = self.bound)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometricBound {
    pub polynomial: SymPolynomial,
    pub geo_params: Vec<SymExpr>,
}

impl GeometricBound {
    pub fn zero(n: usize) -> Self {
        GeometricBound {
            polynomial: SymPolynomial::zero(),
            geo_params: vec![SymExpr::zero(); n],
        }
    }

    pub fn marginalize(&self, var: Var) -> Self {
        let mut polynomial = self.polynomial.clone();
        let mut geo_params = self.geo_params.clone();
        polynomial =
            polynomial.marginalize(var) * (SymExpr::one() - geo_params[var.id()].clone()).inverse();
        geo_params[var.id()] = SymExpr::zero();
        Self {
            polynomial,
            geo_params,
        }
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> GeometricBound {
        Self {
            polynomial: self.polynomial.substitute(replacements),
            geo_params: self
                .geo_params
                .iter()
                .map(|p| p.substitute(replacements))
                .collect(),
        }
    }

    pub fn evaluate_var<T: From<f64> + Number>(
        &self,
        inputs: &[T],
        var: Var,
        degree_p1: usize,
    ) -> TaylorPoly<T> {
        let vars = inputs
            .iter()
            .enumerate()
            .map(|(w, val)| {
                if w == var.id() {
                    TaylorPoly::var(var, val.clone(), degree_p1)
                } else {
                    TaylorPoly::from(val.clone())
                }
            })
            .collect::<Vec<_>>();
        self.eval(&vars)
    }

    pub fn eval<T: From<f64> + Number>(&self, inputs: &[TaylorPoly<T>]) -> TaylorPoly<T> {
        let numerator = self.polynomial.eval(inputs);
        let mut denominator = TaylorPoly::one();
        for (v, geo_param) in self.geo_params.iter().enumerate() {
            denominator *= TaylorPoly::one()
                - TaylorPoly::from(T::from(geo_param.extract_constant().unwrap()))
                    * inputs[v].clone();
        }
        numerator / denominator
    }

    pub fn total_mass(&self) -> SymExpr {
        let numer = self
            .polynomial
            .eval_expr(&vec![1.0; self.polynomial.num_vars()]);
        let mut denom = SymExpr::one();
        for geo_param in &self.geo_params {
            denom *= SymExpr::one() - geo_param.clone();
        }
        numer / denom
    }
}

impl std::fmt::Display for GeometricBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.polynomial)?;
        writeln!(f, "______________________________________________________")?;
        for (i, param) in self.geo_params.iter().enumerate() {
            write!(f, "(1 - {param} * {})", Var(i))?;
        }
        writeln!(f)
    }
}

impl std::ops::Mul<SymExpr> for GeometricBound {
    type Output = Self;

    fn mul(mut self, rhs: SymExpr) -> Self::Output {
        self.polynomial *= rhs;
        self
    }
}

impl std::ops::Div<SymExpr> for GeometricBound {
    type Output = Self;

    fn div(mut self, rhs: SymExpr) -> Self::Output {
        self.polynomial /= rhs;
        self
    }
}
