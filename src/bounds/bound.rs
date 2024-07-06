use crate::{
    bounds::sym_expr::SymExpr,
    multivariate_taylor::TaylorPoly,
    number::{Number, Rational},
    ppl::Var,
    semantics::support::VarSupport,
    support::SupportSet,
};
use ndarray::{ArrayD, ArrayViewD, Axis, Slice};
use num_traits::{One, Zero};
use std::ops::AddAssign;

use super::float_rat::FloatRat;

#[derive(Debug, Clone)]
pub struct BoundResult {
    pub lower: FiniteDiscrete,
    pub upper: GeometricBound,
    pub var_supports: VarSupport,
}

impl BoundResult {
    pub fn zero(n: usize) -> BoundResult {
        BoundResult {
            lower: FiniteDiscrete::zero(n),
            upper: GeometricBound::zero(n),
            var_supports: VarSupport::empty(n),
        }
    }
    pub fn marginalize(self, var: Var) -> BoundResult {
        let mut var_supports = self.var_supports;
        if !var_supports[var].is_empty() {
            var_supports.set(var, SupportSet::zero());
        }
        BoundResult {
            lower: self.lower.marginalize(var),
            upper: self.upper.marginalize(var),
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
        writeln!(f, "Lower bound:\n{bound}", bound = self.lower)?;
        writeln!(f, "Upper bound:\n{bound}", bound = self.upper)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteDiscrete {
    pub masses: ArrayD<Rational>,
}

impl FiniteDiscrete {
    pub fn zero(n: usize) -> FiniteDiscrete {
        FiniteDiscrete {
            masses: ArrayD::zeros(vec![1; n]),
        }
    }

    pub fn marginalize(&self, Var(v): Var) -> FiniteDiscrete {
        let masses = self.masses.sum_axis(Axis(v)).insert_axis(Axis(v));
        FiniteDiscrete { masses }
    }

    pub fn total_mass(&self) -> Rational {
        self.masses.sum()
    }

    pub fn extend_axis(&mut self, Var(v): Var, new_len: usize) {
        let axis = Axis(v);
        let old_len = self.masses.len_of(axis);
        if new_len <= old_len {
            return;
        }
        let mut new_shape = self.masses.shape().to_owned();
        new_shape[v] = new_len - old_len;
        self.masses
            .append(axis, ArrayD::zeros(new_shape).view())
            .unwrap();
    }

    pub fn shift_left(&mut self, Var(v): Var, offset: usize) {
        let len = self.masses.len_of(Axis(v));
        let zero_elem = self
            .masses
            .slice_axis(Axis(v), Slice::from(0..=offset.min(len - 1)))
            .sum_axis(Axis(v));
        self.masses
            .slice_axis_inplace(Axis(v), Slice::from(offset.min(len - 1)..));
        self.masses.index_axis_mut(Axis(v), 0).assign(&zero_elem);
    }

    pub fn shift_right(&mut self, Var(v): Var, offset: usize) {
        let mut zero_shape = self.masses.shape().to_owned();
        zero_shape[v] = offset;
        self.masses = ndarray::concatenate(
            Axis(v),
            &[ArrayD::zeros(zero_shape).view(), self.masses.view()],
        )
        .unwrap();
    }

    pub fn add_categorical(&mut self, Var(v): Var, categorical: &[Rational]) {
        let len = self.masses.len_of(Axis(v));
        let max = categorical.len() - 1;
        self.extend_axis(Var(v), len + max);
        let mut new_shape = self.masses.shape().to_owned();
        new_shape[v] = len + max;
        let masses = self.masses.clone();
        self.masses = ArrayD::zeros(new_shape);
        for (offset, prob) in categorical.iter().enumerate() {
            if prob > &Rational::zero() {
                self.masses
                    .slice_axis_mut(Axis(v), Slice::from(offset..len + offset))
                    .add_assign(
                        &masses
                            .slice_axis(Axis(v), Slice::from(0..len))
                            .map(|e| e.clone() * prob.clone()),
                    );
            }
        }
    }

    pub fn probs(&self, Var(v): Var) -> Vec<Rational> {
        self.masses.axis_iter(Axis(v)).map(|ms| ms.sum()).collect()
    }

    pub fn moments(&self, Var(v): Var, n: usize) -> Vec<Rational> {
        let probs = self.probs(Var(v));
        let mut moments = vec![Rational::zero(); n];
        for (k, p) in probs.into_iter().enumerate() {
            let mut power = p;
            for moment in &mut moments {
                *moment += power.clone();
                power *= Rational::from_int(k);
            }
        }
        moments
    }
}

impl std::fmt::Display for FiniteDiscrete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.masses.map(Rational::round_to_f64).fmt(f)
    }
}

impl std::ops::AddAssign<&Self> for FiniteDiscrete {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.masses.ndim(), rhs.masses.ndim());
        if self.masses.shape() == rhs.masses.shape() {
            self.masses += &rhs.masses;
        } else {
            let lhs_shape = self.masses.shape();
            let rhs_shape = rhs.masses.shape();
            let shape = lhs_shape
                .iter()
                .zip(rhs_shape)
                .map(|(&l, &r)| l.max(r))
                .collect::<Vec<_>>();
            let mut masses = ArrayD::zeros(shape);
            masses
                .slice_each_axis_mut(|ax| (0..lhs_shape[ax.axis.index()]).into())
                .add_assign(&self.masses);
            masses
                .slice_each_axis_mut(|ax| (0..rhs_shape[ax.axis.index()]).into())
                .add_assign(&rhs.masses);
            self.masses = masses;
        }
    }
}

impl std::ops::Add for &FiniteDiscrete {
    type Output = FiniteDiscrete;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.masses.ndim(), rhs.masses.ndim());
        let lhs_shape = self.masses.shape();
        let rhs_shape = rhs.masses.shape();
        let shape = lhs_shape
            .iter()
            .zip(rhs_shape)
            .map(|(&l, &r)| l.max(r))
            .collect::<Vec<_>>();
        let mut masses = ArrayD::zeros(shape);
        masses
            .slice_each_axis_mut(|ax| (0..lhs_shape[ax.axis.index()]).into())
            .add_assign(&self.masses);
        masses
            .slice_each_axis_mut(|ax| (0..rhs_shape[ax.axis.index()]).into())
            .add_assign(&rhs.masses);
        FiniteDiscrete { masses }
    }
}

impl std::ops::MulAssign<Rational> for FiniteDiscrete {
    fn mul_assign(&mut self, rhs: Rational) {
        for elem in &mut self.masses {
            *elem *= rhs.clone();
        }
    }
}

impl std::ops::Mul<Rational> for FiniteDiscrete {
    type Output = Self;

    fn mul(mut self, rhs: Rational) -> Self::Output {
        self *= rhs;
        self
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
    pub masses: ArrayD<SymExpr>,
    pub geo_params: Vec<SymExpr>,
}

impl GeometricBound {
    pub fn zero(n: usize) -> Self {
        GeometricBound {
            masses: ArrayD::zeros(vec![1; n]),
            geo_params: vec![SymExpr::zero(); n],
        }
    }

    pub fn mass(&self, mut idx: Vec<usize>) -> SymExpr {
        assert_eq!(idx.len(), self.masses.ndim());
        let mut factor = SymExpr::one();
        for (v, i) in idx.iter_mut().enumerate() {
            let len = self.masses.len_of(Axis(v));
            if *i >= len {
                factor *= self.geo_params[v].clone().pow((*i - len + 1) as i32);
                *i = len - 1;
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

    pub fn shift_left(&mut self, Var(v): Var, offset: usize) {
        self.extend_axis(Var(v), offset + 2);
        let zero_elem = self
            .masses
            .slice_axis(Axis(v), Slice::from(0..=offset))
            .sum_axis(Axis(v));
        self.masses
            .slice_axis_inplace(Axis(v), Slice::from(offset..));
        self.masses.index_axis_mut(Axis(v), 0).assign(&zero_elem);
    }

    pub fn shift_right(&mut self, Var(v): Var, offset: usize) {
        let mut zero_shape = self.masses.shape().to_owned();
        zero_shape[v] = offset;
        self.masses = ndarray::concatenate(
            Axis(v),
            &[ArrayD::zeros(zero_shape).view(), self.masses.view()],
        )
        .unwrap();
    }

    pub fn add_categorical(&mut self, Var(v): Var, categorical: &[Rational]) {
        let len = self.masses.len_of(Axis(v));
        let max = categorical.len() - 1;
        self.extend_axis(Var(v), len + max);
        let mut new_shape = self.masses.shape().to_owned();
        new_shape[v] = len + max;
        let masses = self.masses.clone();
        self.masses = ArrayD::zeros(new_shape);
        for (offset, prob) in categorical.iter().enumerate() {
            if prob > &Rational::zero() {
                self.masses
                    .slice_axis_mut(Axis(v), Slice::from(offset..len + max))
                    .add_assign(
                        &masses
                            .slice_axis(Axis(v), Slice::from(0..len + max - offset))
                            .map(|e| e.clone() * SymExpr::from(prob.clone())),
                    );
            }
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

    pub fn resolve(&self, assignments: &[Rational]) -> GeometricBound {
        let masses = self
            .masses
            .map(|c| SymExpr::from(c.eval_exact(assignments)));
        let geo_params = self
            .geo_params
            .iter()
            .map(|p| SymExpr::from(p.eval_exact(assignments)))
            .collect();
        Self { masses, geo_params }
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> GeometricBound {
        Self {
            masses: self.masses.map(|c| c.substitute(replacements)),
            geo_params: self
                .geo_params
                .iter()
                .map(|p| p.substitute(replacements))
                .collect(),
        }
    }

    pub fn eval_taylor<T: From<FloatRat> + Number>(
        &self,
        inputs: &[TaylorPoly<T>],
    ) -> TaylorPoly<T> {
        Self::eval_taylor_impl(&self.masses.view(), &self.geo_params, inputs)
    }

    pub fn eval_taylor_impl<T: From<FloatRat> + Number>(
        coeffs: &ArrayViewD<SymExpr>,
        geo_params: &[SymExpr],
        inputs: &[TaylorPoly<T>],
    ) -> TaylorPoly<T> {
        if coeffs.ndim() == 0 {
            return TaylorPoly::from(T::from(coeffs[[]].extract_constant().unwrap().clone()));
        }
        let len = coeffs.len_of(Axis(0));
        let denominator = TaylorPoly::one()
            - TaylorPoly::from(T::from(geo_params[0].extract_constant().unwrap().clone()))
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

    pub fn eval_expr(&self, inputs: &[Rational]) -> SymExpr {
        Self::eval_expr_impl(&self.masses.view(), &self.geo_params, inputs)
    }

    fn eval_expr_impl(
        coeffs: &ArrayViewD<SymExpr>,
        geo_params: &[SymExpr],
        inputs: &[Rational],
    ) -> SymExpr {
        let nvars = coeffs.ndim();
        if nvars == 0 {
            return coeffs.first().unwrap().clone();
        }
        let len = coeffs.len_of(Axis(0));
        let denominator = SymExpr::one() - geo_params[0].clone() * SymExpr::from(inputs[0].clone());
        let mut res = Self::eval_expr_impl(
            &coeffs.index_axis(Axis(0), len - 1),
            &geo_params[1..],
            &inputs[1..],
        );
        res /= denominator;
        for subview in coeffs.axis_iter(Axis(0)).rev().skip(1) {
            res *= SymExpr::from(inputs[0].clone());
            res += Self::eval_expr_impl(&subview, &geo_params[1..], &inputs[1..]);
        }
        res
    }

    pub fn total_mass(&self) -> SymExpr {
        self.eval_expr(&vec![Rational::one(); self.masses.ndim()])
    }

    pub fn expected_value(&self, Var(v): Var) -> SymExpr {
        let len = self.masses.len_of(Axis(v));
        let mut rest_params = self.geo_params.clone();
        let alpha = rest_params.remove(v);
        let mut res = Self::eval_expr_impl(
            &self.masses.index_axis(Axis(v), len - 1),
            &rest_params,
            &vec![Rational::one(); rest_params.len()],
        );
        let alpha_comp = SymExpr::one() - alpha.clone();
        res *= (alpha / alpha_comp.clone() + SymExpr::from(Rational::from_int(len as u32 - 1)))
            / alpha_comp;
        for (n, subview) in self.masses.axis_iter(Axis(v)).enumerate().rev().skip(1) {
            res += Self::eval_expr_impl(
                &subview,
                &rest_params,
                &vec![Rational::one(); rest_params.len()],
            ) * SymExpr::from(Rational::from_int(n as u32));
        }
        res
    }

    pub fn tail_objective(&self, v: Var) -> SymExpr {
        (SymExpr::one() - self.geo_params[v.id()].clone()).inverse()
    }
}

impl std::fmt::Display for GeometricBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EGD(\n{},", self.masses)?;
        write!(f, "[")?;
        for param in &self.geo_params {
            write!(f, "{param}, ")?;
        }
        writeln!(f, "])")
    }
}

impl std::ops::Mul<SymExpr> for GeometricBound {
    type Output = Self;

    fn mul(mut self, rhs: SymExpr) -> Self::Output {
        for elem in &mut self.masses {
            *elem *= rhs.clone();
        }
        self
    }
}

impl std::ops::MulAssign<SymExpr> for GeometricBound {
    fn mul_assign(&mut self, rhs: SymExpr) {
        for elem in &mut self.masses {
            *elem *= rhs.clone();
        }
    }
}

impl std::ops::Div<SymExpr> for GeometricBound {
    type Output = Self;

    fn div(mut self, rhs: SymExpr) -> Self::Output {
        for elem in &mut self.masses {
            *elem /= rhs.clone();
        }
        self
    }
}

impl std::ops::DivAssign<SymExpr> for GeometricBound {
    fn div_assign(&mut self, rhs: SymExpr) {
        for elem in &mut self.masses {
            *elem /= rhs.clone();
        }
    }
}
