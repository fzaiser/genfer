use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{One, Zero};

use crate::bounds::sym_expr::SymExpr;

use super::float_rat::FloatRat;

#[derive(Clone, Debug)]
pub struct LinearExpr {
    pub coeffs: Vec<FloatRat>,
    pub constant: FloatRat,
}

impl LinearExpr {
    pub fn new(coeffs: Vec<FloatRat>, constant: FloatRat) -> Self {
        Self { coeffs, constant }
    }

    pub fn zero() -> Self {
        Self::new(vec![], FloatRat::zero())
    }

    pub fn one() -> Self {
        Self::new(vec![FloatRat::one()], FloatRat::zero())
    }

    pub fn constant(constant: FloatRat) -> Self {
        Self::new(vec![], constant)
    }

    pub fn var(var: usize) -> Self {
        let mut coeffs = vec![FloatRat::zero(); var + 1];
        coeffs[var] = FloatRat::one();
        Self::new(coeffs, FloatRat::zero())
    }

    pub fn as_constant(&self) -> Option<&FloatRat> {
        if self.coeffs.iter().all(Zero::is_zero) {
            Some(&self.constant)
        } else {
            None
        }
    }

    pub fn to_lp_expr(
        &self,
        vars: &[good_lp::Variable],
        conv: &impl Fn(&FloatRat) -> f64,
    ) -> good_lp::Expression {
        let mut result = good_lp::Expression::from(conv(&self.constant));
        for (coeff, var) in self.coeffs.iter().zip(vars) {
            result.add_mul(conv(coeff), var);
        }
        result
    }

    pub fn grad_norm(&self) -> f64 {
        self.coeffs
            .iter()
            .map(|c| c.float() * c.float())
            .sum::<f64>()
            .sqrt()
    }

    pub fn normalize(&self) -> Self {
        let grad_len = self.grad_norm();
        if grad_len == 0.0 {
            return self.clone();
        }
        let grad_len = FloatRat::from_f64(grad_len);
        let coeffs = self
            .coeffs
            .iter()
            .map(|c| c.clone() / grad_len.clone())
            .collect();
        let constant = self.constant.clone() / grad_len;
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
            } else if *coeff == -FloatRat::one() {
                write!(f, "-{}", SymExpr::var(i))?;
            } else {
                write!(f, "{}{}", coeff, SymExpr::var(i))?;
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

impl Mul<FloatRat> for LinearExpr {
    type Output = Self;

    #[inline]
    fn mul(self, other: FloatRat) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c * other.clone()).collect(),
            self.constant * other.clone(),
        )
    }
}

impl Div<FloatRat> for LinearExpr {
    type Output = Self;

    #[inline]
    fn div(self, other: FloatRat) -> Self::Output {
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
    pub fn eq(e1: LinearExpr, e2: LinearExpr) -> Self
    where
        LinearExpr: Sub<Output = LinearExpr>,
    {
        Self {
            expr: e2 - e1,
            eq_zero: true,
        }
    }

    pub fn le(e1: LinearExpr, e2: LinearExpr) -> Self
    where
        LinearExpr: Sub<Output = LinearExpr>,
    {
        Self {
            expr: e2 - e1,
            eq_zero: false,
        }
    }

    pub fn to_lp_constraint(
        &self,
        var_list: &[good_lp::Variable],
        conv: &impl Fn(&FloatRat) -> f64,
    ) -> good_lp::Constraint {
        let result = self.expr.to_lp_expr(var_list, conv);
        if self.eq_zero {
            result.eq(0.0)
        } else {
            result.geq(0.0)
        }
    }

    pub fn eval_constant(&self) -> Option<bool> {
        let constant = self.expr.as_constant()?;
        if self.eq_zero {
            Some(constant.is_zero())
        } else {
            Some(constant >= &FloatRat::zero())
        }
    }

    pub fn normalize(&self) -> Self {
        Self {
            expr: self.expr.normalize(),
            eq_zero: self.eq_zero,
        }
    }

    pub fn tighten(&self, eps: f64) -> Self {
        let constant = if self.expr.grad_norm() == 0.0 {
            self.expr.constant.clone()
        } else {
            self.expr.constant.clone() - FloatRat::from_f64(eps)
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
