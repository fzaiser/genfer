#![allow(clippy::needless_range_loop)]

use std::ops::{AddAssign, SubAssign};

use ndarray::{indices, prelude::*, Slice};
use num_traits::{One, Zero};

use crate::{number::Number, ppl::Var};

#[derive(Clone, PartialEq, Eq)]
/// A multivariate Taylor polynomial.
// TODO: wrap this into an Rc to make cloning cheaper?
pub struct TaylorPoly<T> {
    coeffs: ArrayD<T>,
    /// Degrees plus one
    ///
    /// These are the "conceptual" degrees of the polynomial, which may be higher than the shape of `coeffs`.
    degrees_p1: Vec<usize>,
}

impl<T> TaylorPoly<T> {
    #[inline]
    fn check_invariants(&self) {
        debug_assert!(self.coeffs.ndim() == self.degrees_p1.len());
        debug_assert!(self
            .coeffs
            .shape()
            .iter()
            .zip(&self.degrees_p1)
            .all(|(s, d)| 0 < *s && s <= d));
    }

    pub fn new(coeffs: ArrayD<T>, degrees_p1: Vec<usize>) -> Self {
        debug_assert!(coeffs.ndim() == degrees_p1.len());
        debug_assert!(coeffs
            .shape()
            .iter()
            .zip(&degrees_p1)
            .all(|(s, d)| 0 < *s && s <= d));
        Self { coeffs, degrees_p1 }
    }

    pub fn from_coeffs(coeffs: ArrayD<T>) -> Self {
        let shape = coeffs.shape().to_vec();
        Self::new(coeffs, shape)
    }

    pub fn num_vars(&self) -> usize {
        self.check_invariants();
        self.degrees_p1.len()
    }

    pub fn shape(&self) -> &[usize] {
        self.check_invariants();
        &self.degrees_p1
    }

    pub fn array(&self) -> &ArrayD<T> {
        self.check_invariants();
        &self.coeffs
    }

    pub fn into_array(self) -> ArrayD<T> {
        self.check_invariants();
        self.coeffs
    }

    pub fn len_of(&self, Var(v): Var) -> usize {
        self.check_invariants();
        if v < self.degrees_p1.len() {
            self.degrees_p1[v]
        } else {
            usize::MAX
        }
    }

    pub fn extend_to_dim(mut self, ndim: usize, degree_p1: usize) -> Self {
        self.check_invariants();
        debug_assert!(self.coeffs.ndim() <= ndim);
        for i in self.coeffs.ndim()..ndim {
            self.coeffs.insert_axis_inplace(Axis(i));
        }
        self.degrees_p1.resize(ndim, degree_p1);
        self
    }

    #[cfg(test)]
    fn extend(mut self, new_size: &[usize]) -> Self
    where
        T: Clone + Zero,
    {
        self.check_invariants();
        debug_assert!(self.degrees_p1.len() <= new_size.len());
        debug_assert!(self
            .coeffs
            .shape()
            .iter()
            .zip(new_size)
            .all(|(s, d)| s <= d));
        for i in self.coeffs.ndim()..new_size.len() {
            self.coeffs.insert_axis_inplace(Axis(i));
        }
        let mut new_coeffs = ArrayD::zeros(new_size);
        new_coeffs
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        Self::new(new_coeffs, new_size.to_vec())
    }

    fn min_degrees_p1(&self, other: &Self) -> Vec<usize> {
        self.check_invariants();
        other.check_invariants();
        let mut degrees_p1 = vec![usize::MAX; self.degrees_p1.len().max(other.degrees_p1.len())];
        for v in 0..degrees_p1.len() {
            if v < self.degrees_p1.len() {
                degrees_p1[v] = degrees_p1[v].min(self.degrees_p1[v]);
            }
            if v < other.degrees_p1.len() {
                degrees_p1[v] = degrees_p1[v].min(other.degrees_p1[v]);
            }
        }
        degrees_p1
    }

    fn max_shape(&self, other: &Self) -> Vec<usize> {
        self.check_invariants();
        other.check_invariants();
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for v in 0..shape.len() {
            if v < self.coeffs.ndim() {
                shape[v] = shape[v].max(self.coeffs.len_of(Axis(v)));
            }
            if v < other.coeffs.ndim() {
                shape[v] = shape[v].max(other.coeffs.len_of(Axis(v)));
            }
            if v < self.degrees_p1.len() {
                shape[v] = shape[v].min(self.degrees_p1[v]);
            }
            if v < other.degrees_p1.len() {
                shape[v] = shape[v].min(other.degrees_p1[v]);
            }
        }
        shape
    }

    fn sum_shape(&self, other: &Self) -> Vec<usize> {
        self.check_invariants();
        other.check_invariants();
        let mut shape = vec![0; self.coeffs.ndim().max(other.coeffs.ndim())];
        for v in 0..shape.len() {
            if v < self.coeffs.ndim() {
                shape[v] += self.coeffs.len_of(Axis(v)) - 1;
            }
            if v < other.coeffs.ndim() {
                shape[v] += other.coeffs.len_of(Axis(v)) - 1;
            }
            shape[v] += 1;
            if v < self.degrees_p1.len() {
                shape[v] = shape[v].min(self.degrees_p1[v]);
            }
            if v < other.degrees_p1.len() {
                shape[v] = shape[v].min(other.degrees_p1[v]);
            }
        }
        shape
    }

    pub fn remove_last_variable(mut self) -> TaylorPoly<T> {
        self.check_invariants();
        let v = self.num_vars() - 1;
        if v < self.coeffs.ndim() {
            self.coeffs.index_axis_inplace(Axis(v), 0);
        }
        let mut degrees_p1 = self.degrees_p1;
        let _ = degrees_p1.pop();
        Self::new(self.coeffs, degrees_p1)
    }

    pub fn truncate_to_degree_p1(mut self, degree_p1: usize) -> TaylorPoly<T> {
        self.check_invariants();
        for v in 0..self.num_vars() {
            self.degrees_p1[v] = self.degrees_p1[v].min(degree_p1);
            if v < self.coeffs.ndim() && self.coeffs.len_of(Axis(v)) > degree_p1 {
                self.coeffs
                    .slice_axis_inplace(Axis(v), Slice::from(0..degree_p1));
            }
        }
        self
    }

    fn truncate_degrees_p1(&mut self, degrees_p1: &[usize]) {
        self.check_invariants();
        for v in 0..self.num_vars() {
            self.degrees_p1[v] = self.degrees_p1[v].min(degrees_p1[v]);
            if v < self.coeffs.ndim() && self.coeffs.len_of(Axis(v)) > degrees_p1[v] {
                self.coeffs
                    .slice_axis_inplace(Axis(v), Slice::from(0..degrees_p1[v]));
            }
        }
    }
}

impl<T: Number> TaylorPoly<T> {
    pub fn zero_with(degrees_p1: Vec<usize>) -> Self {
        Self::new(
            arr0(T::zero())
                .into_dyn()
                .into_shape(vec![1; degrees_p1.len()])
                .unwrap(),
            degrees_p1,
        )
    }

    /// Construct a constant Taylor polynomial.
    pub fn from_u32(constant: u32) -> Self {
        Self::from_u32_with(constant, vec![])
    }

    pub fn from_u32_with(constant: u32, degrees_p1: Vec<usize>) -> Self {
        Self::new(arr0(T::from(constant)).into_dyn(), degrees_p1)
    }

    /// Construct the Taylor polynomial 0 + 1 * v + 0 * v^2 + ... + 0 * v^(len - 1).
    pub fn var_at_zero(v: Var, len: usize) -> Self {
        let v = v.id();
        let mut shape = vec![1; v + 1];
        shape[v] = 2;
        let mut coeffs = ArrayD::zeros(shape);
        if len > 1 {
            *coeffs.index_axis_mut(Axis(v), 1).first_mut().unwrap() = T::one();
        }
        Self::new(coeffs, vec![len; v + 1])
    }

    pub fn var(Var(v): Var, x: T, len: usize) -> Self {
        let mut shape = vec![1; v + 1];
        shape[v] = 2;
        let mut coeffs = ArrayD::zeros(shape);
        *coeffs.index_axis_mut(Axis(v), 0).first_mut().unwrap() = x;
        if len > 1 {
            *coeffs.index_axis_mut(Axis(v), 1).first_mut().unwrap() = T::one();
        }
        Self::new(coeffs, vec![len; v + 1])
    }

    pub fn var_with_degrees_p1(Var(v): Var, x: T, degrees_p1: Vec<usize>) -> Self {
        let mut shape = vec![1; degrees_p1.len()];
        shape[v] = 2;
        let mut coeffs = ArrayD::zeros(shape);
        *coeffs.index_axis_mut(Axis(v), 0).first_mut().unwrap() = x;
        if degrees_p1[v] > 1 {
            *coeffs.index_axis_mut(Axis(v), 1).first_mut().unwrap() = T::one();
        }
        Self::new(coeffs, degrees_p1)
    }

    /// If the Taylor polynomial is constant, return the constant.
    pub fn extract_constant(&self) -> Option<T> {
        self.check_invariants();
        if self.coeffs.len() == 1 {
            self.coeffs.first().cloned()
        } else {
            None
        }
    }

    /// If self is a linear polynomial in some variable, return the coefficients and variable.
    ///
    /// This function does not recognize constants.
    #[inline]
    fn extract_linear(&self) -> Option<(T, T, Var)> {
        self.check_invariants();
        for v in 0..self.coeffs.ndim() {
            if self.coeffs.len_of(Axis(v)) < 2 {
                continue;
            }
            if self.coeffs.axis_iter(Axis(v)).enumerate().all(|(i, view)| {
                if i <= 1 {
                    view.iter().skip(1).all(Zero::is_zero)
                } else {
                    view.iter().all(Zero::is_zero)
                }
            }) {
                let constant = self.coeffs.index_axis(Axis(v), 0).first().unwrap().clone();
                let factor = self.coeffs.index_axis(Axis(v), 1).first().unwrap().clone();
                return Some((constant, factor, Var(v)));
            }
        }
        None
    }

    pub fn constant_term(&self) -> T {
        self.check_invariants();
        self.coeffs.first().unwrap().clone()
    }

    pub fn zero_pad(&mut self, new_degrees_p1: Vec<usize>) {
        self.check_invariants();
        debug_assert!(self
            .degrees_p1
            .iter()
            .zip(&new_degrees_p1)
            .all(|(&a, &b)| a <= b));
        self.degrees_p1 = new_degrees_p1;
        for i in self.coeffs.ndim()..self.degrees_p1.len() {
            self.coeffs.insert_axis_inplace(Axis(i));
        }
    }

    pub fn coefficient(&self, index: &[usize]) -> T {
        self.check_invariants();
        let mut view = self.coeffs.view();
        for (v, &idx) in index.iter().enumerate() {
            assert!(
                idx < self.len_of(Var(v)),
                "index out of bounds: the index is {index:?} but the size is {:?}",
                self.degrees_p1
            );
            if v >= self.coeffs.ndim() {
                if idx != 0 {
                    return T::zero();
                }
            } else if idx >= self.coeffs.len_of(Axis(v)) {
                return T::zero();
            } else {
                view.index_axis_inplace(Axis(0), idx);
            }
        }
        assert!(
            view.ndim() == 0,
            "index is too short: the index is {index:?} but the size is {:?}",
            self.degrees_p1
        );
        view.first().unwrap().clone()
    }

    pub fn coefficients_of_term(&self, Var(v): Var, order: usize) -> Self {
        self.check_invariants();
        if v >= self.coeffs.ndim() {
            if order == 0 {
                return self.clone();
            }
            return Self::zero_with(self.degrees_p1.clone());
        }
        if order >= self.coeffs.len_of(Axis(v)) {
            return Self::zero_with(self.degrees_p1.clone());
        }
        Self::new(
            self.coeffs
                .slice_axis(Axis(v), Slice::from(order..=order))
                .to_owned(),
            self.degrees_p1.clone(),
        )
    }

    pub fn taylor_polynomial(&self, Var(v): Var, order: usize) -> TaylorPoly<T> {
        self.check_invariants();
        assert!(v < self.num_vars() && order < self.len_of(Var(v)));
        if v >= self.coeffs.ndim() {
            if order == 0 {
                return self.clone();
            }
            return Self::zero_with(self.degrees_p1.clone());
        }
        if order >= self.coeffs.len_of(Axis(v)) {
            return self.clone();
        }
        let upper = self.coeffs.len_of(Axis(v)).min(order + 1);
        let result = self
            .coeffs
            .slice_axis(Axis(v), Slice::from(0..upper))
            .to_owned();
        Self::new(result, self.degrees_p1.clone())
    }

    pub fn taylor_polynomial_terms(&self, Var(v): Var, orders: &[usize]) -> TaylorPoly<T> {
        self.check_invariants();
        let max_order_p1 = orders.iter().max().map_or(1, |m| m + 1);
        if v >= self.coeffs.ndim() {
            if orders.contains(&0) {
                return self.clone();
            }
            return Self::zero_with(self.degrees_p1.clone());
        }
        let upper = self.coeffs.len_of(Axis(v)).min(max_order_p1);
        let mut result = self
            .coeffs
            .slice_axis(Axis(v), Slice::from(0..upper))
            .to_owned();
        let mut keep_term = vec![false; max_order_p1];
        for order in orders {
            keep_term[*order] = true;
        }
        for i in 0..upper {
            if !keep_term[i] {
                result.index_axis_mut(Axis(v), i).fill(T::zero());
            }
        }
        Self::new(result, self.degrees_p1.clone())
    }

    pub fn exp(&self) -> Self {
        self.check_invariants();
        let mut result_shape = self.degrees_p1.clone();
        for i in 0..result_shape.len() {
            if self.coeffs.len_of(Axis(i)) == 1 {
                result_shape[i] = 1;
            }
        }
        let mut result = ArrayD::zeros(result_shape);
        exp(&self.coeffs.view(), &mut result.view_mut());
        Self::new(result, self.degrees_p1.clone())
    }

    pub fn log(&self) -> Self {
        self.check_invariants();
        let mut result_shape = self.degrees_p1.clone();
        for i in 0..result_shape.len() {
            if self.coeffs.len_of(Axis(i)) == 1 {
                result_shape[i] = 1;
            }
        }
        let mut result = ArrayD::zeros(result_shape);
        log(&self.coeffs.view(), &mut result.view_mut());
        Self::new(result, self.degrees_p1.clone())
    }

    /// Binary exponentiation
    pub fn pow(&self, mut exp: u32) -> Self {
        self.check_invariants();
        if exp == 0 {
            return Self::one();
        }
        if exp == 1 {
            return self.clone();
        }
        let mut res = Self::one();
        let mut base = self.clone();
        while exp > 0 {
            if exp & 1 == 1 {
                res *= base.clone();
            }
            base *= base.clone();
            exp >>= 1;
        }
        res
    }

    /// Returns the `n`-th derivative of this Taylor expansion.
    ///
    /// Compared with [`taylor_expansion_of_coeff`], it has an additional factor of `n`!.
    /// This can prevent underflow sometimes, which makes this function useful.
    pub fn derivative(&self, Var(v): Var, n: usize) -> Self {
        self.check_invariants();
        assert!(v < self.num_vars() && n < self.len_of(Var(v)));
        if v >= self.coeffs.ndim() {
            if n == 0 {
                return self.clone();
            }
            return Self::zero_with(self.degrees_p1.clone());
        }
        let mut degrees_p1 = self.degrees_p1.clone();
        degrees_p1[v] = degrees_p1[v].saturating_sub(n);
        if n >= self.coeffs.len_of(Axis(v)) {
            return Self::zero_with(degrees_p1);
        }
        let mut result = self.coeffs.slice_axis(Axis(v), Slice::from(n..)).to_owned();
        let mut falling_factorial = (1..=n).fold(T::one(), |acc, i| acc * T::from(i as u32));
        result
            .axis_iter_mut(Axis(v))
            .enumerate()
            .for_each(|(k, mut res)| {
                res.map_mut(|x| *x *= falling_factorial.clone());
                falling_factorial *= T::from((n + k + 1) as u32) / T::from((k + 1) as u32);
            });
        Self::new(result, degrees_p1)
    }

    /// Returns the Taylor expansion of the `n`-the coefficient of this Taylor expansion.
    pub fn taylor_expansion_of_coeff(&self, Var(v): Var, n: usize) -> Self {
        self.check_invariants();
        assert!(v < self.num_vars() && n < self.len_of(Var(v)));
        if v >= self.coeffs.ndim() {
            if n == 0 {
                return self.clone();
            }
            return Self::zero_with(self.degrees_p1.clone());
        }
        let mut degrees_p1 = self.degrees_p1.clone();
        degrees_p1[v] = degrees_p1[v].saturating_sub(n);
        if n >= self.coeffs.len_of(Axis(v)) {
            return Self::zero_with(degrees_p1);
        }
        let mut result = self.coeffs.slice_axis(Axis(v), Slice::from(n..)).to_owned();
        let mut factor = T::one();
        result
            .axis_iter_mut(Axis(v))
            .enumerate()
            .skip(1)
            .for_each(|(k, mut res)| {
                factor *= T::from((n + k) as u32) / T::from(k as u32);
                res.map_mut(|x| *x *= factor.clone());
            });
        Self::new(result, degrees_p1)
    }

    /// Shift the coefficients of `v` down by `n`, accumulating the ones that would be pushed out at zero.
    ///
    /// For example shifting `2 + 3 * v + v^2` down by 1 yields `5 + v`.
    pub fn shift_down(&self, Var(v): Var, n: usize) -> Self {
        self.check_invariants();
        assert!(v < self.num_vars() && n < self.len_of(Var(v)));
        if v >= self.coeffs.ndim() {
            return self.clone();
        }
        let mut degrees_p1 = self.degrees_p1.clone();
        degrees_p1[v] = degrees_p1[v].saturating_sub(n);
        let result = if self.coeffs.len_of(Axis(v)) <= n + 1 {
            let result = self.coeffs.sum_axis(Axis(v)).to_owned();
            result.insert_axis(Axis(v))
        } else {
            let mut result = self.coeffs.slice_axis(Axis(v), Slice::from(n..)).to_owned();
            result.index_axis_mut(Axis(v), 0).add_assign(
                &self
                    .coeffs
                    .slice_axis(Axis(v), Slice::from(..n))
                    .sum_axis(Axis(v)),
            );
            result
        };
        Self::new(result, degrees_p1)
    }

    /// Substitutes a Taylor expansion for a variable of this Taylor expansion.
    /// The substitution must have the same order as this Taylor expansion.
    pub fn subst_var(&self, Var(v): Var, subst: &Self) -> Self {
        self.check_invariants();
        subst.check_invariants();
        if v >= self.coeffs.ndim() {
            return self.clone();
        }
        let degrees_p1 = self.min_degrees_p1(subst);
        if subst.is_zero() {
            return Self::new(
                self.coeffs
                    .slice_axis(Axis(v), Slice::from(0..=0))
                    .to_owned(),
                degrees_p1,
            );
        }
        if let Some((c, m, Var(w))) = subst.extract_linear() {
            if v == w && c.is_zero() {
                let mut factor = T::one();
                let mut result = self
                    .coeffs
                    .slice_each_axis(|ax| Slice::from(0..ax.len.min(degrees_p1[ax.axis.index()])))
                    .to_owned();
                for mut coeff in result.axis_iter_mut(Axis(v)) {
                    coeff.map_inplace(|x| *x *= factor.clone());
                    factor *= m.clone();
                }
                return Self::new(result, degrees_p1);
            }
        }
        let mut res = Self::zero_with(degrees_p1.clone());
        let mut coeffs = self.coeffs.to_owned();
        for i in coeffs.ndim()..degrees_p1.len() {
            coeffs.insert_axis_inplace(Axis(i));
        }
        for coeff in coeffs.axis_chunks_iter(Axis(v), 1).rev() {
            let coeff =
                coeff.slice_each_axis(|ax| Slice::from(0..ax.len.min(degrees_p1[ax.axis.index()])));
            res = res * subst.clone() + Self::new(coeff.to_owned(), degrees_p1.clone());
        }
        res
    }

    #[inline]
    pub fn evaluate_all_one(&self) -> T {
        self.check_invariants();
        self.coeffs.iter().fold(T::zero(), |acc, x| acc + x.clone())
    }

    #[inline]
    pub fn mul_var(
        mut self,
        m: T,
        Var(v): Var,
        shape: Vec<usize>,
        degrees_p1: Vec<usize>,
    ) -> TaylorPoly<T> {
        let upper = (shape[v] - 1).min(self.coeffs.len_of(Axis(v)));
        self.coeffs.slice_axis_inplace(Axis(v), (..upper).into());
        self.coeffs.map_inplace(|x| *x *= m.clone());
        let mut result = ArrayD::zeros(shape.clone());
        result
            .slice_axis_mut(Axis(v), Slice::from(1..=upper))
            .assign(
                &self
                    .coeffs
                    .slice_each_axis(|ax| Slice::from(0..ax.len.min(shape[ax.axis.index()]))),
            );
        Self::new(result, degrees_p1)
    }

    #[inline]
    pub fn mul_linear(
        self,
        c: T,
        m: T,
        var: Var,
        shape: Vec<usize>,
        degrees_p1: Vec<usize>,
    ) -> TaylorPoly<T> {
        if c.is_zero() {
            return self.mul_var(m, var, shape, degrees_p1);
        }
        self.clone().mul_var(m, var, shape, degrees_p1) + self * TaylorPoly::from(c)
    }
}

impl<T: Clone> From<T> for TaylorPoly<T> {
    fn from(x: T) -> Self {
        Self::from_coeffs(arr0(x).into_dyn())
    }
}

impl<T: std::fmt::Display> std::fmt::Debug for TaylorPoly<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TaylorPoly({:?}, {})", self.degrees_p1, self.coeffs)
    }
}

impl<T: Number> Zero for TaylorPoly<T> {
    fn zero() -> Self {
        Self::from(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs.first().unwrap().is_zero()
    }
}

impl<T: Number> One for TaylorPoly<T> {
    fn one() -> Self {
        Self::from(T::one())
    }

    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs.first().unwrap().is_one()
    }
}

#[macro_export]
macro_rules! taylor {
    ([$($a:expr$(,)*)*]) => {
        {
            use $crate::number::F64;
            use ndarray::{array, ArrayD};
            let arr: ArrayD<f64> = (array![$(($a),)*]).into_dyn();
            TaylorPoly::from_coeffs(arr.map(|x| F64::from(*x)))
        }
    };
    ($a:expr) => {
        {
            use $crate::number::F64;
            use ndarray::ArrayD;
            let arr: ArrayD<f64> = arr0($a).into_dyn();
            TaylorPoly::from_coeffs(arr.map(|x| F64::from(*x)))
        }
    };
    ([$($a:expr$(,)*)*]; $b:tt) => {
        {
            use $crate::number::F64;
            use ndarray::{array, ArrayD};
            let arr: ArrayD<f64> = array![$(($a),)*].into_dyn();
            TaylorPoly::new(arr.map(|x| F64::from(*x)), vec!$b)
        }
    };
    ($a:expr; $b:tt) => {
        {
            use $crate::number::F64;
            use ndarray::ArrayD;
            let arr: ArrayD<f64> = arr0($a).into_dyn();
            TaylorPoly::new(arr.map(|x| F64::from(*x)), vec!$b)
        }
    };
}

pub fn fmt_polynomial<T: std::fmt::Display + Zero>(
    f: &mut std::fmt::Formatter<'_>,
    coeffs: &ArrayViewD<T>,
) -> std::fmt::Result {
    let shape = coeffs.shape();
    let mut first = true;
    for index in indices(shape) {
        if coeffs[&index].is_zero() {
            continue;
        }
        if first {
            first = false;
        } else {
            write!(f, " + ")?;
        }
        write!(f, "{}", coeffs[&index])?;
        for (i, exponent) in index.as_array_view().into_iter().enumerate() {
            if *exponent == 0 {
                continue;
            }
            write!(f, "{}", Var(i))?;
            if *exponent > 1 {
                write!(f, "^{exponent}")?;
            }
        }
    }
    if first {
        write!(f, "0")?;
    }
    Ok(())
}

impl<T: std::fmt::Display + Zero + One> std::fmt::Display for TaylorPoly<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_polynomial(f, &self.coeffs.view())
    }
}

#[test]
pub fn test_2d_derivative() {
    let taylor = taylor!([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ]);
    assert_eq!(
        taylor.derivative(Var(0), 1),
        taylor!([
            [5.0, 6.0, 7.0, 8.0],
            [18.0, 20.0, 22.0, 24.0],
            [39.0, 42.0, 45.0, 48.0]
        ])
    );
    assert_eq!(
        taylor.derivative(Var(1), 1),
        taylor!([
            [2.0, 6.0, 12.0],
            [6.0, 14.0, 24.0],
            [10.0, 22.0, 36.0],
            [14.0, 30.0, 48.0]
        ])
    );
    assert_eq!(
        taylor.derivative(Var(0), 2),
        taylor.derivative(Var(0), 1).derivative(Var(0), 1)
    );
    assert_eq!(
        taylor.derivative(Var(1), 2),
        taylor.derivative(Var(1), 1).derivative(Var(1), 1)
    );
    assert_eq!(
        taylor.derivative(Var(0), 3),
        taylor
            .derivative(Var(0), 1)
            .derivative(Var(0), 1)
            .derivative(Var(0), 1)
    );
}

#[test]
pub fn test_2d_taylor_expansion_of_coeff() {
    let taylor = taylor!([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ]);
    assert_eq!(
        taylor.taylor_expansion_of_coeff(Var(0), 2),
        taylor!([[9.0, 10.0, 11.0, 12.0], [39.0, 42.0, 45.0, 48.0]])
    );
    assert_eq!(
        taylor.taylor_expansion_of_coeff(Var(1), 3),
        taylor!([[4.0], [8.0], [12.0], [16.0]])
    );
    let expected = taylor!([[11.0, 36.0], [45.0, 144.0]]);
    assert_eq!(
        taylor
            .taylor_expansion_of_coeff(Var(0), 2)
            .taylor_expansion_of_coeff(Var(1), 2),
        expected
    );
    assert_eq!(
        taylor
            .taylor_expansion_of_coeff(Var(1), 2)
            .taylor_expansion_of_coeff(Var(0), 2),
        expected
    );
}

#[test]
fn test_2d_subst_var() {
    let taylor = taylor!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],]);
    let subst = taylor!([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0],]);
    assert_eq!(
        taylor.subst_var(Var(0), &subst),
        taylor!([
            [741.0, 2436.0, 5353.0],
            [1872.0, 6163.0, 13516.0],
            [3487.0, 11452.0, 25030.0]
        ])
    );
    assert_eq!(
        taylor.subst_var(Var(1), &subst),
        taylor!([
            [321.0, 682.0, 1107.0],
            [1460.0, 3101.0, 5016.0],
            [4111.0, 8736.0, 14088.0]
        ])
    );
    assert_ne!(
        taylor.subst_var(Var(0), &subst).subst_var(Var(1), &subst),
        taylor.subst_var(Var(1), &subst).subst_var(Var(0), &subst),
    )
}

#[inline]
fn broadcast<T: Number>(xs: &mut TaylorPoly<T>, ys: &mut TaylorPoly<T>) {
    xs.check_invariants();
    ys.check_invariants();
    if xs.degrees_p1.len() < ys.degrees_p1.len() {
        xs.degrees_p1
            .extend(ys.degrees_p1.iter().skip(xs.degrees_p1.len()));
    } else if ys.degrees_p1.len() < xs.degrees_p1.len() {
        ys.degrees_p1
            .extend(xs.degrees_p1.iter().skip(ys.degrees_p1.len()));
    }
    if xs.coeffs.ndim() < ys.coeffs.ndim() {
        for i in xs.coeffs.ndim()..ys.coeffs.ndim() {
            xs.coeffs.insert_axis_inplace(Axis(i));
        }
    }
    if ys.coeffs.ndim() < xs.coeffs.ndim() {
        for i in ys.coeffs.ndim()..xs.coeffs.ndim() {
            ys.coeffs.insert_axis_inplace(Axis(i));
        }
    }
}

impl<T: Number> std::ops::Add for TaylorPoly<T> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self {
        let result_degrees_p1 = self.min_degrees_p1(&other);
        broadcast(&mut self, &mut other);
        self.truncate_degrees_p1(&result_degrees_p1);
        other.truncate_degrees_p1(&result_degrees_p1);
        if other.coeffs.len() == 1 {
            *self.coeffs.first_mut().unwrap() += other.coeffs.first().unwrap().clone();
            return Self::new(self.coeffs, result_degrees_p1);
        }
        if self.coeffs.len() == 1 {
            *other.coeffs.first_mut().unwrap() += self.coeffs.first().unwrap().clone();
            return Self::new(other.coeffs, result_degrees_p1);
        }
        let shape = self.max_shape(&other);
        self.truncate_degrees_p1(&shape);
        other.truncate_degrees_p1(&shape);
        let mut result = ArrayD::zeros(shape.clone());
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .add_assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .add_assign(&other.coeffs);
        Self::new(result, result_degrees_p1)
    }
}

#[test]
fn test_add_mismatched_shapes() {
    let a = TaylorPoly::var(Var(0), crate::number::F64::one(), 5);
    let b = TaylorPoly::var(Var(1), crate::number::F64::one(), 4);
    assert_eq!(
        (a.clone() + b.clone()).extend(&[5, 4]),
        a.extend(&[5, 4]) + b.extend(&[5, 4])
    );
}

impl<T: Number> std::ops::AddAssign for TaylorPoly<T> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        // TODO: this could be optimized to work in-place.
        *self = self.clone() + other;
    }
}

impl<T: Number> std::ops::Neg for TaylorPoly<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.coeffs, self.degrees_p1)
    }
}

impl<T: Number> std::ops::Sub for TaylorPoly<T> {
    type Output = Self;

    fn sub(mut self, mut other: Self) -> Self {
        let result_degrees_p1 = self.min_degrees_p1(&other);
        broadcast(&mut self, &mut other);
        self.truncate_degrees_p1(&result_degrees_p1);
        other.truncate_degrees_p1(&result_degrees_p1);
        if other.coeffs.len() == 1 {
            *self.coeffs.first_mut().unwrap() -= other.coeffs.first().unwrap().clone();
            return Self::new(self.coeffs, result_degrees_p1);
        }
        if self.coeffs.len() == 1 {
            *other.coeffs.first_mut().unwrap() -= self.coeffs.first().unwrap().clone();
            return Self::new(-other.coeffs, result_degrees_p1);
        }
        let shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(shape);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .add_assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .sub_assign(&other.coeffs);
        Self::new(result, result_degrees_p1)
    }
}

#[test]
fn test_sub_mismatched_shapes() {
    let a = TaylorPoly::var(Var(0), crate::number::F64::one(), 5);
    let b = TaylorPoly::var(Var(1), crate::number::F64::one(), 4);
    assert_eq!(
        (a.clone() - b.clone()).extend(&[5, 4]),
        a.extend(&[5, 4]) - b.extend(&[5, 4])
    );
}

impl<T: Number> std::ops::SubAssign for TaylorPoly<T> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        // TODO: this could be optimized to work in-place.
        *self = self.clone() - other;
    }
}

#[inline]
pub(crate) fn extract_1d_len(shape: &[usize]) -> Option<usize> {
    let mut res = None;
    for len in shape {
        if *len != 1 {
            if res.is_some() {
                return None;
            }
            res = Some(*len);
        }
    }
    res
}

#[inline]
fn mul_1d<T: Number>(xs: Vec<T>, ys: Vec<T>, n: usize) -> Vec<T> {
    let mut zs = vec![T::zero(); n];
    for k in 0..n {
        let lo = (k + 1).saturating_sub(ys.len());
        let hi = (k + 1).min(xs.len());
        for j in lo..hi {
            zs[k] += xs[j].clone() * ys[k - j].clone();
        }
    }
    zs
}

fn mul<T: Number>(xs: &ArrayViewD<T>, ys: &ArrayViewD<T>, res: &mut ArrayViewMutD<T>) {
    if res.is_empty() {
        return;
    }
    if res.ndim() == 0 {
        *res.first_mut().unwrap() += xs.first().unwrap().clone() * ys.first().unwrap().clone();
        return;
    }
    if let Some(n) = extract_1d_len(res.shape()) {
        let out = mul_1d(
            xs.iter().cloned().collect::<Vec<_>>(),
            ys.iter().cloned().collect::<Vec<_>>(),
            n,
        );
        res.iter_mut().zip(out).for_each(|(z, o)| *z += o);
        return;
    }
    for (k, mut z) in res.axis_iter_mut(Axis(0)).enumerate() {
        let lo = (k + 1).saturating_sub(ys.len_of(Axis(0)));
        let hi = (k + 1).min(xs.len_of(Axis(0)));
        for j in lo..hi {
            mul(
                &xs.index_axis(Axis(0), j),
                &ys.index_axis(Axis(0), k - j),
                &mut z,
            );
        }
    }
}

impl<T: Number> std::ops::Mul for TaylorPoly<T> {
    type Output = Self;

    fn mul(mut self, mut other: Self) -> Self {
        let degrees_p1 = self.min_degrees_p1(&other);

        // Recognize multiplication by zero:
        if self.is_zero() || other.is_zero() {
            return Self::zero_with(degrees_p1);
        }

        // Broadcast to common shape:
        broadcast(&mut self, &mut other);
        let shape = self.sum_shape(&other);
        self.truncate_degrees_p1(&degrees_p1);
        other.truncate_degrees_p1(&degrees_p1);

        // Recognize multiplication by one:
        if self.is_one() {
            return other;
        }
        if other.is_one() {
            return self;
        }

        // Recognize multiplication by a constant:
        if let Some(c) = self.extract_constant() {
            other.coeffs.map_inplace(|x| *x = c.clone() * x.clone());
            return other;
        }
        if let Some(c) = other.extract_constant() {
            self.coeffs.map_inplace(|x| *x = c.clone() * x.clone());
            return self;
        }

        // Recognize multiplication by c*v for a variable v:
        // Multiplication with these happens when we substitute p*v in V ~ Binomial(X, p),
        // so they are quite common and worth optimizing for.
        if let Some((c, m, v)) = self.extract_linear() {
            let mut shape = other.coeffs.shape().to_vec();
            shape[v.id()] = degrees_p1[v.id()].min(shape[v.id()] + 1);
            return other.mul_linear(c, m, v, shape, degrees_p1);
        }
        if let Some((c, m, v)) = other.extract_linear() {
            let mut shape = self.coeffs.shape().to_vec();
            shape[v.id()] = degrees_p1[v.id()].min(shape[v.id()] + 1);
            return self.mul_linear(c, m, v, shape, degrees_p1);
        }

        // General case:
        let mut result = ArrayD::zeros(shape);
        mul(
            &self.coeffs.view(),
            &other.coeffs.view(),
            &mut result.view_mut(),
        );
        Self::new(result, degrees_p1)
    }
}

impl<T: Number> std::ops::MulAssign for TaylorPoly<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

#[test]
fn test_mul_mismatched_shapes() {
    let a = TaylorPoly::var(Var(0), crate::number::F64::one(), 5);
    let b = TaylorPoly::var(Var(1), crate::number::F64::one(), 4);
    assert_eq!(
        (a.clone() * b.clone()).extend(&[5, 4]),
        a.clone().extend(&[5, 4]) * b.clone().extend(&[5, 4])
    );
    let c = a.clone() * a.clone() * a;
    let d = b.clone() * b;
    assert_eq!(
        (c.clone() * d.clone()).extend(&[5, 4]),
        c.extend(&[5, 4]) * d.extend(&[5, 4])
    );
}

#[test]
fn test_2d_mul() {
    let f = taylor!([[1.0, 2.0,], [3.0, 4.0]]);
    let g = taylor!([[5.0, 6.0], [7.0, 8.0]]);
    let result = f * g;
    assert_eq!(result, taylor!([[5.0, 16.0], [22.0, 60.0]]));
}

#[test]
fn test_2d_mul_const() {
    let f = taylor!([[1.0, 2.0,], [3.0, 4.0]]);
    let g = taylor!([[5.0, 6.0], [7.0, 8.0]]);
    assert_eq!(f.clone() * g, taylor!([[5.0, 16.0], [22.0, 60.0]]));
    assert_eq!(
        f.clone() * TaylorPoly::zero(),
        TaylorPoly::zero_with(vec![2, 2])
    );
    assert_eq!(
        TaylorPoly::zero() * f.clone(),
        TaylorPoly::zero_with(vec![2, 2])
    );
    assert_eq!(f.clone() * TaylorPoly::one(), f.clone());
    assert_eq!(TaylorPoly::one() * f.clone(), f.clone());
    assert_eq!(
        TaylorPoly::from_u32(2) * f.clone(),
        taylor!([[2.0, 4.0], [6.0, 8.0]])
    );
    assert_eq!(
        f.clone() * TaylorPoly::from_u32(2),
        taylor!([[2.0, 4.0], [6.0, 8.0]])
    );
}

#[test]
fn test_2d_mul_factor_linear() {
    let f = taylor!([[1.0, 2.0,], [3.0, 4.0]]);
    let g0 = TaylorPoly::from_u32(2) * TaylorPoly::var_at_zero(Var(0), 2);
    assert_eq!(
        g0.extract_linear(),
        Some((Zero::zero(), 2.0.into(), Var(0)))
    );
    let g1 = TaylorPoly::from_u32(3) * TaylorPoly::var_at_zero(Var(1), 2);
    assert_eq!(
        g1.extract_linear(),
        Some((Zero::zero(), 3.0.into(), Var(1)))
    );
    assert_eq!(f.clone() * g0.clone(), taylor!([[0.0, 0.0], [2.0, 4.0]]));
    assert_eq!(f.clone() * g1.clone(), taylor!([[0.0, 3.0], [0.0, 9.0]]));
    assert_eq!(g0.clone() * f.clone(), taylor!([[0.0, 0.0], [2.0, 4.0]]));
    assert_eq!(g1.clone() * f.clone(), taylor!([[0.0, 3.0], [0.0, 9.0]]));
    assert_eq!(g0.clone() * g1.clone(), taylor!([[0.0, 0.0], [0.0, 6.0]]));
    assert_eq!(g1.clone() * g0.clone(), taylor!([[0.0, 0.0], [0.0, 6.0]]));

    let f = taylor!([[1.0, 2.0,], [3.0, 4.0]]);
    let g0 = taylor!([3.0, 2.0]);
    assert_eq!(g0.extract_linear(), Some((3.0.into(), 2.0.into(), Var(0))));
    let g1 = taylor!([[3.0, 2.0], [0.0, 0.0]]);
    assert_eq!(g1.extract_linear(), Some((3.0.into(), 2.0.into(), Var(1))));
    assert_eq!(f.clone() * g0.clone(), taylor!([[3.0, 6.0], [11.0, 16.0]]));
    assert_eq!(f.clone() * g1.clone(), taylor!([[3.0, 8.0], [9.0, 18.0]]));
    assert_eq!(g0.clone() * f.clone(), taylor!([[3.0, 6.0], [11.0, 16.0]]));
    assert_eq!(g1.clone() * f.clone(), taylor!([[3.0, 8.0], [9.0, 18.0]]));
    assert_eq!(g0.clone() * g1.clone(), taylor!([[9.0, 6.0], [6.0, 4.0]]));
    assert_eq!(g1.clone() * g0.clone(), taylor!([[9.0, 6.0], [6.0, 4.0]]));
}

fn div<T: Number>(xs: &ArrayViewD<T>, ys: &ArrayViewD<T>, res: &mut ArrayViewMutD<T>) {
    if xs.is_empty() {
        return;
    }
    if res.ndim() == 0 {
        *res.first_mut().unwrap() = xs.first().unwrap().clone() / ys.first().unwrap().clone();
        return;
    }
    for k in 0..res.len_of(Axis(0)) {
        let (previous, mut current) = res.view_mut().split_at(Axis(0), k);
        let mut current = current.index_axis_mut(Axis(0), 0);
        let lo = (k + 1).saturating_sub(ys.len_of(Axis(0)));
        for j in lo..k {
            mul(
                &previous.index_axis(Axis(0), j),
                &ys.index_axis(Axis(0), k - j),
                &mut current,
            );
        }
        current.map_inplace(|x| *x = -x.clone());
        if k < xs.len_of(Axis(0)) {
            let xs_k = xs.index_axis(Axis(0), k);
            current
                .slice_each_axis_mut(|ax| Slice::from(0..xs_k.len_of(ax.axis)))
                .add_assign(&xs_k);
        }
        let copy = current.to_owned();
        current.fill(T::zero());
        div(&copy.view(), &ys.index_axis(Axis(0), 0), &mut current);
    }
}

impl<T: Number> std::ops::Div for TaylorPoly<T> {
    type Output = Self;

    fn div(mut self, mut other: Self) -> Self {
        // Broadcast to common shape:
        broadcast(&mut self, &mut other);
        let degrees_p1 = self.min_degrees_p1(&other);
        self.truncate_degrees_p1(&degrees_p1);
        other.truncate_degrees_p1(&degrees_p1);

        // Recognize division by one:
        if other.is_one() {
            return self;
        }

        // Recognize division by a constant:
        if let Some(c) = other.extract_constant() {
            self.coeffs.map_inplace(|x| *x = x.clone() / c.clone());
            return self;
        }

        // General case:
        let mut result_shape = degrees_p1.clone();
        for i in 0..result_shape.len() {
            if other.coeffs.len_of(Axis(i)) == 1 {
                result_shape[i] = self.coeffs.len_of(Axis(i));
            }
        }
        let mut result = ArrayD::zeros(result_shape);

        div(
            &self.coeffs.view(),
            &other.coeffs.view(),
            &mut result.view_mut(),
        );
        Self::new(result, degrees_p1)
    }
}

impl<T: Number> std::ops::DivAssign for TaylorPoly<T> {
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}

#[test]
fn test_div_mismatched_shapes() {
    let a = TaylorPoly::var(Var(0), crate::number::F64::one(), 5);
    let b = TaylorPoly::var(Var(1), crate::number::F64::one(), 4);
    assert_eq!(
        (a.clone() / b.clone()).extend(&[5, 4]),
        a.clone().extend(&[5, 4]) / b.clone().extend(&[5, 4])
    );
    let c = a.clone() * a.clone() * a;
    let d = b.clone() * b;
    assert_eq!(
        (c.clone() * d.clone()).extend(&[5, 4]),
        c.extend(&[5, 4]) * d.extend(&[5, 4])
    );
}

#[test]
fn test_2d_div() {
    let f = taylor!([[1.0, 2.0,], [3.0, 4.0]]);
    let g = taylor!([[5.0, 6.0], [7.0, 8.0]]);
    let result = f.clone() / g.clone();
    assert_eq!(
        result,
        taylor!([
            [0.2, 0.159_999_999_999_999_98],
            [0.319_999_999_999_999_95, -0.127_999_999_999_999_9]
        ])
    );
    assert_eq!(result * g, f);
}

#[inline]
fn exp_1d<T: Number>(xs: Vec<T>, n: usize) -> Vec<T> {
    let mut res = vec![T::zero(); n];
    res[0] = xs[0].exp();
    for k in 1..n {
        let mut sum = T::zero();
        let hi = xs.len().min(k + 1);
        for j in 1..hi {
            sum += xs[j].clone() * T::from(j as u32) * res[k - j].clone();
        }
        res[k] = sum / T::from(k as u32);
    }
    res
}

fn exp<T: Number>(xs: &ArrayViewD<T>, res: &mut ArrayViewMutD<T>) {
    if xs.is_empty() {
        return;
    }
    if res.ndim() == 0 {
        *res.first_mut().unwrap() = xs.first().unwrap().exp();
        return;
    }
    if let Some(n) = extract_1d_len(res.shape()) {
        let out = exp_1d(xs.iter().cloned().collect::<Vec<_>>(), n);
        res.iter_mut().zip(out).for_each(|(z, o)| *z = o);
        return;
    }
    exp(
        &xs.index_axis(Axis(0), 0),
        &mut res.index_axis_mut(Axis(0), 0),
    );
    for k in 1..res.len_of(Axis(0)) {
        let (previous, mut current) = res.view_mut().split_at(Axis(0), k);
        let mut current = current.index_axis_mut(Axis(0), 0);
        let hi = xs.len_of(Axis(0)).min(k + 1);
        for j in 1..hi {
            mul(
                &xs.index_axis(Axis(0), j)
                    .map(|x| x.clone() * T::from(j as u32))
                    .view(),
                &previous.index_axis(Axis(0), k - j),
                &mut current,
            );
        }
        current.map_mut(|x| *x /= T::from(k as u32));
    }
}

fn log_1d<T: Number>(xs: Vec<T>, n: usize) -> Vec<T> {
    let mut res = vec![T::zero(); n];
    res[0] = xs[0].log();
    for k in 1..n {
        let mut sum = T::zero();
        let lo = (k + 1).saturating_sub(xs.len()).max(1);
        for j in lo..k {
            sum += xs[k - j].clone() * res[j].clone() * T::from(j as u32);
        }
        res[k] = (xs.get(k).cloned().unwrap_or_else(T::zero) * T::from(k as u32) - sum)
            / xs[0].clone()
            / T::from(k as u32);
    }
    res
}

fn log<T: Number>(xs: &ArrayViewD<T>, res: &mut ArrayViewMutD<T>) {
    if xs.is_empty() {
        return;
    }
    if res.ndim() == 0 {
        *res.first_mut().unwrap() = xs.first().unwrap().log();
        return;
    }
    if extract_1d_len(xs.shape()).is_some() {
        let out = log_1d(
            xs.iter().cloned().collect::<Vec<_>>(),
            extract_1d_len(res.shape()).unwrap(),
        );
        res.iter_mut().zip(out).for_each(|(z, o)| *z = o);
        return;
    }
    log(
        &xs.index_axis(Axis(0), 0),
        &mut res.index_axis_mut(Axis(0), 0),
    );
    for k in 1..res.len_of(Axis(0)) {
        let (previous, mut current) = res.view_mut().split_at(Axis(0), k);
        let mut current = current.index_axis_mut(Axis(0), 0);
        let lo = (k + 1).saturating_sub(xs.len_of(Axis(0))).max(1);
        for j in lo..k {
            mul(
                &xs.index_axis(Axis(0), k - j),
                &previous
                    .index_axis(Axis(0), j)
                    .map(|x| x.clone() * T::from(j as u32))
                    .view(),
                &mut current,
            );
        }
        current.map_mut(|x| *x = -x.clone());
        if k < xs.len_of(Axis(0)) {
            let xs_k = xs.index_axis(Axis(0), k);
            current
                .slice_each_axis_mut(|ax| Slice::from(0..xs_k.len_of(ax.axis)))
                .add_assign(&xs_k.map(|x| T::from(k as u32) * x.clone()).view());
        }
        current.assign(
            (TaylorPoly::new(current.to_owned(), current.shape().to_vec())
                / TaylorPoly::new(
                    xs.index_axis(Axis(0), 0).to_owned(),
                    current.shape().to_vec(),
                ))
            .array(),
        );
        current.map_mut(|x| *x /= T::from(k as u32));
    }
}

#[test]
fn test_exp_mismatched_shapes() {
    let a = TaylorPoly::var(Var(0), crate::number::F64::one(), 5);
    assert_eq!(a.exp().extend(&[5, 4]), a.clone().extend(&[5, 4]).exp());
    let c = a.clone() * a.clone() * a;
    assert_eq!(c.clone().exp().extend(&[5, 4]), c.extend(&[5, 4]).exp());
    let a = taylor!([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]];
        [5, 4]
    );
    assert_eq!(a.exp().extend(&[5, 4]), a.clone().extend(&[5, 4]).exp());
    let c = a.clone() * a.clone() * a;
    assert_eq!(c.clone().exp().extend(&[5, 4]), c.extend(&[5, 4]).exp());
}

#[test]
fn test_2d_exp() {
    use crate::number::F64;
    assert_eq!(TaylorPoly::<F64>::zero().exp(), TaylorPoly::one());
    let f = taylor!([[1.0, 2.0,], [3.0, 4.0]]);
    let g = taylor!([[5.0, 6.0], [7.0, 8.0]]);
    let result = f.exp();
    assert_eq!(
        result,
        taylor!([
            [2.718_281_828_459_045, 5.436_563_656_918_09],
            [8.154_845_485_377_136, 27.182_818_284_590_454]
        ])
    );
    assert_eq!(
        f.exp() * (-f.clone()).exp(),
        taylor!([[1.0, 0.0], [0.0, 0.0]])
    );
    assert_eq!(
        f.exp() * g.exp(),
        taylor!([
            [403.428_793_492_735_1, 3_227.430_347_941_881],
            [4_034.287_934_927_350_8, 37_115.449_001_331_624]
        ])
    );
    assert_eq!(
        (f + g).exp(),
        taylor!([
            [403.428_793_492_735_1, 3_227.430_347_941_881],
            [4_034.287_934_927_351, 37_115.449_001_331_63]
        ])
    )
}

#[test]
fn test_log_mismatched_shapes() {
    let a = TaylorPoly::var(Var(0), crate::number::F64::one(), 5);
    assert_eq!(a.log().extend(&[5, 4]), a.clone().extend(&[5, 4]).log());
    let c = a.clone() * a.clone() * a;
    assert_eq!(c.clone().log().extend(&[5, 4]), c.extend(&[5, 4]).log());
    let a = taylor!([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]];
        [5, 4]
    );
    assert_eq!(a.log().extend(&[5, 4]), a.clone().extend(&[5, 4]).log());
    let c = a.clone() * a.clone() * a;
    assert_eq!(c.clone().log().extend(&[5, 4]), c.extend(&[5, 4]).log());
}

#[test]
fn test_2d_log() {
    use crate::number::F64;
    assert_eq!(TaylorPoly::<F64>::one().log(), TaylorPoly::zero());
    let xp1 = TaylorPoly::var(Var(0), F64::one(), 5);
    assert_eq!(
        xp1.log(),
        taylor!([0.0, 1.0, -0.5, 0.333_333_333_333_333_3, -0.25])
    );
    let e = taylor!([1.0, 2.0, 3.0]);
    assert_eq!(e.log(), taylor!([0.0, 2.0, 1.0]));
    assert_eq!(e.log().exp(), e);
    let f = taylor!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let g = taylor!([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0], [9.0, 10.0, 11.0]]);
    assert_eq!(
        f.log(),
        taylor!([[0.0, 2.0, 1.0], [4.0, -3.0, 0.0], [-1.0, 6.0, -4.5]])
    );
    assert_eq!(
        f.log().exp(),
        taylor!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    );
    assert_eq!(
        f.exp().log(),
        taylor!([
            [1.0, 2.0, 3.000000000000001],
            [4.0, 4.999999999999999, 6.000000000000007],
            [6.999999999999999, 8.000000000000002, 8.999999999999991]
        ])
    );
    assert_eq!(
        f.log() + (TaylorPoly::one() / f.clone()).log(),
        taylor!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    );
    assert_eq!(
        f.log() + g.log(),
        taylor!([
            [1.6094379124341003, 3.2, 1.6800000000000002],
            [5.4, -3.0799999999999996, -0.06400000000000006],
            [-0.17999999999999994, 5.952, -4.5416]
        ])
    );
    assert_eq!(
        (f * g).log(),
        taylor!([
            [1.6094379124341003, 3.2, 1.6799999999999997],
            [5.4, -3.080000000000001, -0.06399999999999864],
            [-0.18000000000000113, 5.952000000000003, -4.5416000000000025]
        ])
    );
}
