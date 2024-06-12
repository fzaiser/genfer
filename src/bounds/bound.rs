use crate::{
    bounds::sym_expr::SymExpr, multivariate_taylor::TaylorPoly, number::Number, ppl::Var,
    semantics::support::VarSupport, support::SupportSet,
};
use ndarray::{ArrayD, ArrayViewD, Axis, Slice};
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

/// The generating function of a geometric bound.
///
/// It is represented as follows.
/// Suppose `masses` is the array
/// ```text
/// [[a, b, c],
/// [d, e, f]]
/// ```
/// and `geo_params` is the vector `[p, q]`.
/// Then the generating function is
/// ```text
/// a + b * x_1 + c * x_1^2 / (1 - q * x_1)
/// + d * x_0 / (1 - p * x_0) + e * x_0 * x_1 / (1 - p * x_0) + f * x_0 * x_1^2 / (1 - p * x_0) / (1 - q * x_1)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GeometricBound {
    pub masses: ArrayD<SymExpr<f64>>,
    pub geo_params: Vec<SymExpr<f64>>,
}

impl GeometricBound {
    pub fn zero(n: usize) -> Self {
        GeometricBound {
            masses: ArrayD::zeros(vec![1; n]),
            geo_params: vec![SymExpr::zero(); n],
        }
    }

    pub fn mass(&self, mut idx: Vec<usize>) -> SymExpr<f64> {
        assert_eq!(idx.len(), self.masses.ndim());
        let mut factor = SymExpr::one();
        for v in 0..idx.len() {
            let len = self.masses.len_of(Axis(v));
            let i = idx[v];
            if i >= len {
                factor *= self.geo_params[v].clone().pow((i - len + 1) as i32);
                idx[v] = len - 1;
            }
        }
        self.masses[idx.as_slice()].clone() * factor
    }

    pub fn extend_axis(&mut self, var: Var, new_len: usize) {
        let axis = Axis(var.id());
        let old_len = self.masses.len_of(axis);
        if new_len <= old_len {
            return;
        }
        let mut new_shape = self.masses.shape().to_owned();
        new_shape[var.id()] = new_len - old_len;
        self.masses
            .append(axis, ArrayD::zeros(new_shape).view())
            .unwrap();
        for i in old_len..new_len {
            let (left, mut right) = self.masses.view_mut().split_at(axis, i);
            right.index_axis_mut(axis, 0).assign(
                &left
                    .index_axis(axis, i - 1)
                    .map(|e| e.clone() * self.geo_params[var.id()].clone()),
            );
        }
    }

    pub fn marginalize(&self, var: Var) -> Self {
        let len = self.masses.len_of(Axis(var.id()));
        let axis = Axis(var.id());
        let mut geo_params = self.geo_params.clone();
        let mut masses = self
            .masses
            .slice_axis(axis, Slice::from(len - 1..len))
            .to_owned();
        masses.map_inplace(|e| *e /= SymExpr::one() - geo_params[var.id()].clone());
        for subview in self.masses.axis_chunks_iter(axis, 1).take(len - 1) {
            masses += &subview;
        }
        geo_params[var.id()] = SymExpr::zero();
        Self { masses, geo_params }
    }

    pub fn resolve(&self, assignments: &[f64]) -> GeometricBound {
        let masses = self.masses.map(|c| SymExpr::Constant(c.eval(assignments)));
        let geo_params = self
            .geo_params
            .iter()
            .map(|p| SymExpr::Constant(p.eval(assignments)))
            .collect();
        Self { masses, geo_params }
    }

    pub fn substitute(&self, replacements: &[SymExpr<f64>]) -> GeometricBound {
        Self {
            masses: self.masses.map(|c| c.substitute(replacements)),
            geo_params: self
                .geo_params
                .iter()
                .map(|p| p.substitute(replacements))
                .collect(),
        }
    }

    pub fn eval_taylor<T: From<f64> + Number>(&self, inputs: &[TaylorPoly<T>]) -> TaylorPoly<T> {
        Self::eval_taylor_impl(&self.masses.view(), &self.geo_params, inputs)
    }

    pub fn eval_taylor_impl<T: From<f64> + Number>(
        coeffs: &ArrayViewD<SymExpr<f64>>,
        geo_params: &[SymExpr<f64>],
        inputs: &[TaylorPoly<T>],
    ) -> TaylorPoly<T> {
        if coeffs.ndim() == 0 {
            return TaylorPoly::from(T::from(coeffs[[]].extract_constant().unwrap()));
        }
        let len = coeffs.len_of(Axis(0));
        let denominator = TaylorPoly::one()
            - TaylorPoly::from(T::from(geo_params[0].extract_constant().unwrap()))
                * inputs[0].clone();
        let mut res = Self::eval_taylor_impl(
            &coeffs.index_axis(Axis(0), len - 1),
            &geo_params[1..],
            &inputs[1..],
        );
        res /= denominator;
        for subview in coeffs.axis_iter(Axis(0)).rev().skip(1) {
            res *= inputs[0].clone();
            res += Self::eval_taylor_impl(&subview, &geo_params[1..], &inputs[1..]);
        }
        res
    }

    pub fn eval_expr(&self, inputs: &[f64]) -> SymExpr<f64> {
        Self::eval_expr_impl(&self.masses.view(), &self.geo_params, inputs)
    }

    fn eval_expr_impl(
        coeffs: &ArrayViewD<SymExpr<f64>>,
        geo_params: &[SymExpr<f64>],
        inputs: &[f64],
    ) -> SymExpr<f64> {
        let nvars = coeffs.ndim();
        if nvars == 0 {
            return coeffs.first().unwrap().clone();
        }
        let len = coeffs.len_of(Axis(0));
        let denominator = SymExpr::one() - geo_params[0].clone() * SymExpr::from(inputs[0]);
        let mut res = Self::eval_expr_impl(
            &coeffs.index_axis(Axis(0), len - 1),
            &geo_params[1..],
            &inputs[1..],
        );
        res /= denominator;
        for subview in coeffs.axis_iter(Axis(0)).rev().skip(1) {
            res *= SymExpr::from(inputs[0]);
            res += Self::eval_expr_impl(&subview, &geo_params[1..], &inputs[1..]);
        }
        res
    }

    pub fn total_mass(&self) -> SymExpr<f64> {
        self.eval_expr(&vec![1.0; self.masses.ndim()])
    }

    pub fn expected_value(&self, Var(v): Var) -> SymExpr<f64> {
        let len = self.masses.len_of(Axis(v));
        let mut rest_params = self.geo_params.clone();
        let alpha = rest_params.remove(v);
        let mut res = Self::eval_expr_impl(
            &self.masses.index_axis(Axis(v), len - 1),
            &rest_params,
            &vec![1.0; rest_params.len()],
        );
        let alpha_comp = SymExpr::one() - alpha.clone();
        res *= (alpha / alpha_comp.clone() + SymExpr::from(f64::from(len as u32 - 1))) / alpha_comp;
        for (n, subview) in self.masses.axis_iter(Axis(v)).enumerate().rev().skip(1) {
            res += Self::eval_expr_impl(&subview, &rest_params, &vec![1.0; rest_params.len()])
                * SymExpr::from(f64::from(n as u32));
        }
        res
    }

    pub fn tail_objective(&self, v: Var) -> SymExpr<f64> {
        (SymExpr::one() - self.geo_params[v.id()].clone()).inverse()
    }
}

impl std::fmt::Display for GeometricBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Masses:\n{}", self.masses)?;
        write!(f, "Geometric params: ")?;
        for param in &self.geo_params {
            write!(f, "{param}, ")?;
        }
        writeln!(f)
    }
}

impl std::ops::Mul<SymExpr<f64>> for GeometricBound {
    type Output = Self;

    fn mul(mut self, rhs: SymExpr<f64>) -> Self::Output {
        for elem in &mut self.masses {
            *elem *= rhs.clone();
        }
        self
    }
}

impl std::ops::MulAssign<SymExpr<f64>> for GeometricBound {
    fn mul_assign(&mut self, rhs: SymExpr<f64>) {
        for elem in &mut self.masses {
            *elem *= rhs.clone();
        }
    }
}

impl std::ops::Div<SymExpr<f64>> for GeometricBound {
    type Output = Self;

    fn div(mut self, rhs: SymExpr<f64>) -> Self::Output {
        for elem in &mut self.masses {
            *elem /= rhs.clone();
        }
        self
    }
}

impl std::ops::DivAssign<SymExpr<f64>> for GeometricBound {
    fn div_assign(&mut self, rhs: SymExpr<f64>) {
        for elem in &mut self.masses {
            *elem /= rhs.clone();
        }
    }
}
