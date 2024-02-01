use std::ops::{AddAssign, SubAssign};

use ndarray::{arr0, indices, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dimension, Slice};
use num_traits::{One, Zero};

use crate::{bounds::sym_expr::*, multivariate_taylor::TaylorPoly, number::Number, ppl::Var};

#[derive(Clone, Debug, PartialEq)]
pub struct SymPolynomial {
    pub(crate) coeffs: ArrayD<SymExpr>,
}

impl SymPolynomial {
    #[inline]
    pub fn new(coeffs: ArrayD<SymExpr>) -> Self {
        Self { coeffs }
    }

    pub fn num_vars(&self) -> usize {
        self.coeffs.ndim()
    }

    pub fn var(var: Var) -> Self {
        Self::var_power(var, 1)
    }

    pub fn var_power(Var(v): Var, n: u32) -> SymPolynomial {
        if n == 0 {
            return Self::one();
        }
        let mut shape = vec![1; v + 1];
        shape[v] = n as usize + 1;
        let mut coeffs = ArrayD::zeros(shape);
        *coeffs
            .index_axis_mut(Axis(v), n as usize)
            .first_mut()
            .unwrap() = SymExpr::one();
        Self { coeffs }
    }

    #[inline]
    fn max_shape(&self, other: &Self) -> Vec<usize> {
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for (v, dim) in shape.iter_mut().enumerate() {
            if v < self.coeffs.ndim() {
                *dim = (*dim).max(self.coeffs.len_of(Axis(v)));
            }
            if v < other.coeffs.ndim() {
                *dim = (*dim).max(other.coeffs.len_of(Axis(v)));
            }
        }
        shape
    }

    #[inline]
    fn sum_shape(&self, other: &Self) -> Vec<usize> {
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for (v, dim) in shape.iter_mut().enumerate() {
            if v < self.coeffs.ndim() {
                *dim += self.coeffs.len_of(Axis(v)) - 1;
            }
            if v < other.coeffs.ndim() {
                *dim += other.coeffs.len_of(Axis(v)) - 1;
            }
        }
        shape
    }

    pub fn marginalize(&self, Var(v): Var) -> SymPolynomial {
        let mut result_shape = self.coeffs.shape().to_vec();
        if v >= result_shape.len() {
            return self.clone();
        }
        result_shape[v] = 1;
        let mut result = ArrayD::zeros(result_shape);
        for coeff in self.coeffs.axis_chunks_iter(Axis(v), 1) {
            result += &coeff;
        }
        Self::new(result)
    }

    pub fn coeff_of_var_power(&self, Var(v): Var, order: usize) -> SymPolynomial {
        if v >= self.coeffs.ndim() {
            if order == 0 {
                return self.clone();
            }
            return Self::zero();
        }
        if order >= self.coeffs.len_of(Axis(v)) {
            return Self::zero();
        }
        Self::new(
            self.coeffs
                .slice_axis(Axis(v), Slice::from(order..=order))
                .to_owned(),
        )
    }

    pub fn shift_coeffs_left(&self, Var(v): Var) -> SymPolynomial {
        if v >= self.coeffs.ndim() {
            return SymPolynomial::zero();
        }
        Self::new(self.coeffs.slice_axis(Axis(v), Slice::from(1..)).to_owned())
    }

    pub fn extract_zero_and_shift_left(self, var: Var) -> (SymPolynomial, SymPolynomial) {
        let p0 = self.coeff_of_var_power(var, 0);
        let rest = self - p0.clone();
        let shifted = rest.shift_coeffs_left(var);
        (p0, shifted)
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> SymPolynomial {
        Self::new(self.coeffs.map(|c| c.substitute(replacements)))
    }

    pub fn eval<T: From<f64> + Number>(&self, inputs: &[TaylorPoly<T>]) -> TaylorPoly<T> {
        let coeffs = self
            .coeffs
            .map(|sym_expr| T::from(sym_expr.extract_constant().unwrap()))
            .to_owned();
        let mut taylor = TaylorPoly::new(coeffs, vec![usize::MAX; inputs.len()]);
        for (v, input) in inputs.iter().enumerate() {
            taylor = taylor.subst_var(Var(v), input);
        }
        taylor
    }

    fn eval_expr_impl(array: &ArrayViewD<SymExpr>, points: &[f64]) -> SymExpr {
        let nvars = array.ndim();
        if nvars == 0 {
            return array.first().unwrap().clone();
        }
        let mut res = SymExpr::zero();
        for c in array.axis_iter(Axis(nvars - 1)) {
            res *= SymExpr::from(points[nvars - 1]);
            res += SymPolynomial::eval_expr_impl(&c, points);
        }
        res
    }

    pub fn eval_expr(&self, points: &[f64]) -> SymExpr {
        SymPolynomial::eval_expr_impl(&self.coeffs.view(), points)
    }
}

impl From<f64> for SymPolynomial {
    #[inline]
    fn from(value: f64) -> Self {
        Self::new(arr0(value.into()).into_dyn())
    }
}

impl Zero for SymPolynomial {
    #[inline]
    fn zero() -> Self {
        Self::new(arr0(SymExpr::zero()).into_dyn())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(SymExpr::is_zero)
    }
}

impl One for SymPolynomial {
    #[inline]
    fn one() -> Self {
        Self::new(arr0(SymExpr::one()).into_dyn())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs.first().unwrap().is_one()
    }
}

impl std::ops::Mul<SymExpr> for SymPolynomial {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SymExpr) -> Self::Output {
        Self::new(self.coeffs.mapv(|c| c * rhs.clone()))
    }
}

impl std::ops::MulAssign<SymExpr> for SymPolynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: SymExpr) {
        self.coeffs.mapv_inplace(|c| c * rhs.clone());
    }
}

impl std::ops::Div<SymExpr> for SymPolynomial {
    type Output = Self;

    #[inline]
    fn div(self, rhs: SymExpr) -> Self::Output {
        Self::new(self.coeffs.mapv(|c| c / rhs.clone()))
    }
}

impl std::ops::DivAssign<SymExpr> for SymPolynomial {
    #[inline]
    fn div_assign(&mut self, rhs: SymExpr) {
        self.coeffs.mapv_inplace(|c| c / rhs.clone());
    }
}

impl std::ops::Neg for SymPolynomial {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.coeffs.mapv_inplace(|c| -c);
        self
    }
}

#[inline]
fn broadcast<T>(xs: &mut ArrayD<T>, ys: &mut ArrayD<T>) {
    if xs.ndim() < ys.ndim() {
        for i in xs.ndim()..ys.ndim() {
            xs.insert_axis_inplace(Axis(i));
        }
    }
    if ys.ndim() < xs.ndim() {
        for i in ys.ndim()..xs.ndim() {
            ys.insert_axis_inplace(Axis(i));
        }
    }
}

impl std::ops::Add for SymPolynomial {
    type Output = Self;

    #[inline]
    fn add(mut self, mut other: Self) -> Self::Output {
        let result_shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(result_shape);
        broadcast(&mut self.coeffs, &mut other.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .add_assign(&other.coeffs);
        Self::new(result)
    }
}

impl std::ops::AddAssign for SymPolynomial {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl std::ops::Sub for SymPolynomial {
    type Output = Self;

    #[inline]
    fn sub(mut self, mut other: Self) -> Self::Output {
        let result_shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(result_shape);
        broadcast(&mut self.coeffs, &mut other.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .sub_assign(&other.coeffs);
        Self::new(result)
    }
}

impl std::ops::SubAssign for SymPolynomial {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl std::ops::Mul for SymPolynomial {
    type Output = Self;

    #[inline]
    fn mul(mut self, mut other: Self) -> Self::Output {
        fn mul_helper(
            xs: &ArrayViewD<SymExpr>,
            ys: &ArrayViewD<SymExpr>,
            res: &mut ArrayViewMutD<SymExpr>,
        ) {
            if res.is_empty() {
                return;
            }
            if res.ndim() == 0 {
                *res.first_mut().unwrap() +=
                    xs.first().unwrap().clone() * ys.first().unwrap().clone();
                return;
            }
            for (k, mut z) in res.axis_iter_mut(Axis(0)).enumerate() {
                let lo = (k + 1).saturating_sub(ys.len_of(Axis(0)));
                let hi = (k + 1).min(xs.len_of(Axis(0)));
                for j in lo..hi {
                    mul_helper(
                        &xs.index_axis(Axis(0), j),
                        &ys.index_axis(Axis(0), k - j),
                        &mut z,
                    );
                }
            }
        }

        // Recognize multiplication by zero:
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        // Broadcast to common shape:
        broadcast(&mut self.coeffs, &mut other.coeffs);
        let result_shape = self.sum_shape(&other);

        // Recognize multiplication by one:
        if self.is_one() {
            return other;
        }
        if other.is_one() {
            return self;
        }

        let mut result = ArrayD::zeros(result_shape);
        mul_helper(
            &self.coeffs.view(),
            &other.coeffs.view(),
            &mut result.view_mut(),
        );
        Self::new(result)
    }
}

impl std::ops::MulAssign for SymPolynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl std::fmt::Display for SymPolynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.coeffs.shape();
        let mut first = true;
        let mut empty_output = true;
        for index in indices(shape) {
            if self.coeffs[&index].is_zero() {
                continue;
            }
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.coeffs[&index])?;
            empty_output = false;
            for (i, exponent) in index.as_array_view().into_iter().enumerate() {
                if *exponent == 0 {
                    continue;
                }
                write!(f, "{}", crate::ppl::Var(i))?;
                if *exponent > 1 {
                    write!(f, "^{exponent}")?;
                }
            }
        }
        if empty_output {
            write!(f, "0")?;
        }
        Ok(())
    }
}
