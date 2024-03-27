use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use num_traits::{One, Zero};

use crate::bounds::sym_expr::SymExpr;

#[derive(Clone, Debug)]
pub struct LinearExpr<T> {
    pub coeffs: Vec<T>,
    pub constant: T,
}

impl<T> LinearExpr<T> {
    pub fn new(coeffs: Vec<T>, constant: T) -> Self {
        Self { coeffs, constant }
    }

    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self::new(vec![], T::zero())
    }

    pub fn one() -> Self
    where
        T: Zero + One,
    {
        Self::new(vec![T::one()], T::zero())
    }

    pub fn constant(constant: T) -> Self {
        Self::new(vec![], constant)
    }

    pub fn var(var: usize) -> Self
    where
        T: Clone + Zero + One,
    {
        let mut coeffs = vec![T::zero(); var + 1];
        coeffs[var] = T::one();
        Self::new(coeffs, T::zero())
    }

    pub fn as_constant(&self) -> Option<&T>
    where
        T: Zero,
    {
        if self.coeffs.iter().all(Zero::is_zero) {
            Some(&self.constant)
        } else {
            None
        }
    }

    pub fn to_lp_expr(
        &self,
        vars: &[good_lp::Variable],
        conv: &impl Fn(&T) -> f64,
    ) -> good_lp::Expression {
        let mut result = good_lp::Expression::from(conv(&self.constant));
        for (coeff, var) in self.coeffs.iter().zip(vars) {
            result.add_mul(conv(coeff), var);
        }
        result
    }
}

impl<T: std::fmt::Display> std::fmt::Display for LinearExpr<T>
where
    T: Zero + One + PartialEq + Neg<Output = T>,
{
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
                write!(f, "{}", SymExpr::<T>::var(i))?;
            } else if *coeff == -T::one() {
                write!(f, "-{}", SymExpr::<T>::var(i))?;
            } else {
                write!(f, "{}{}", coeff, SymExpr::<T>::var(i))?;
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

impl<T> Neg for LinearExpr<T>
where
    T: Clone + Neg<Output = T>,
{
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

impl<T> Add for LinearExpr<T>
where
    T: AddAssign,
{
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

impl<T> Sub for LinearExpr<T>
where
    T: Clone + AddAssign + Neg<Output = T>,
{
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<T> Mul<T> for LinearExpr<T>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: T) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c * other.clone()).collect(),
            self.constant * other.clone(),
        )
    }
}

impl<T> Div<T> for LinearExpr<T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    #[inline]
    fn div(self, other: T) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c / other.clone()).collect(),
            self.constant / other.clone(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct LinearConstraint<T> {
    expr: LinearExpr<T>,
    /// If true, `expr` must be equal to zero, otherwise it must be nonnegative
    eq_zero: bool,
}

impl<T> LinearConstraint<T> {
    pub fn eq(e1: LinearExpr<T>, e2: LinearExpr<T>) -> Self
    where
        LinearExpr<T>: Sub<Output = LinearExpr<T>>,
    {
        Self {
            expr: e2 - e1,
            eq_zero: true,
        }
    }

    pub fn le(e1: LinearExpr<T>, e2: LinearExpr<T>) -> Self
    where
        LinearExpr<T>: Sub<Output = LinearExpr<T>>,
    {
        Self {
            expr: e2 - e1,
            eq_zero: false,
        }
    }

    pub fn to_lp_constraint(
        &self,
        var_list: &[good_lp::Variable],
        conv: &impl Fn(&T) -> f64,
    ) -> good_lp::Constraint {
        let result = self.expr.to_lp_expr(var_list, conv);
        if self.eq_zero {
            result.eq(0.0)
        } else {
            result.geq(0.0)
        }
    }

    pub fn eval_constant(&self) -> Option<bool>
    where
        T: Zero + PartialOrd,
    {
        let constant = self.expr.as_constant()?;
        if self.eq_zero {
            Some(constant.is_zero())
        } else {
            Some(constant >= &T::zero())
        }
    }
}

impl<T> std::fmt::Display for LinearConstraint<T>
where
    LinearExpr<T>: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.eq_zero {
            write!(f, "{} = 0", self.expr)
        } else {
            write!(f, "{} >= 0", self.expr)
        }
    }
}
