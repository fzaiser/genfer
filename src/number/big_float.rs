use std::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::{One, Zero};

use super::f64::F64;
use super::number::{FloatNumber, IntervalNumber, Number};

/// Extracts the exponent and sets it to zero
///
/// ```
/// use tool::number::extract_exponent;
/// assert_eq!(extract_exponent(0.0), (0.0, 0));
/// assert_eq!(extract_exponent(1.0), (1.0, 0));
/// assert_eq!(extract_exponent(2.0), (1.0, 1));
/// assert_eq!(extract_exponent(3.0), (1.5, 1));
/// assert_eq!(extract_exponent(0.5), (1.0, -1));
/// assert_eq!(extract_exponent(0.75), (1.5, -1));
/// ```
#[inline]
pub fn extract_exponent(f: f64) -> (f64, i64) {
    if !f.is_finite() {
        return (f, 0);
    }
    if f == 0.0 {
        return (f, 0);
    }
    let bits = f.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let exponent = exponent - 1023;
    if f.is_subnormal() {
        let f = f * f64::powi(2.0, -exponent as i32);
        let bits = f.to_bits();
        let exponent2 = ((bits >> 52) & 0x7ff) as i64;
        let exponent2 = exponent2 - 1023;
        (f * f64::powi(2.0, -exponent2 as i32), exponent + exponent2)
    } else {
        (f * f64::powi(2.0, -exponent as i32), exponent)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BigFloat {
    factor: f64,
    exponent: i64,
}

impl From<BigFloat> for f64 {
    #[inline]
    fn from(x: BigFloat) -> Self {
        x.to_f64()
    }
}

impl BigFloat {
    #[inline]
    pub fn normalize(factor: f64, exponent: i64) -> Self {
        if factor == 0.0 {
            return Self::zero();
        }
        let (factor, e) = extract_exponent(factor);
        let result = Self {
            factor,
            exponent: e.checked_add(exponent).unwrap(),
        };
        debug_assert!(result.is_normalized());
        result
    }

    #[inline]
    fn is_normalized(&self) -> bool {
        if !self.factor.is_finite() || self.factor == 0.0 {
            self.exponent == 0
        } else {
            (self.factor >= 1.0 && self.factor < 2.0) || (self.factor > -2.0 && self.factor <= -1.0)
        }
    }

    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.factor * f64::powi(2.0, self.exponent as i32)
    }
}

impl Zero for BigFloat {
    #[inline]
    fn zero() -> Self {
        Self {
            factor: 0.0,
            exponent: 0,
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.factor == 0.0
    }
}

impl One for BigFloat {
    #[inline]
    fn one() -> Self {
        Self {
            factor: 1.0,
            exponent: 0,
        }
    }
}

impl From<u32> for BigFloat {
    #[inline]
    fn from(u: u32) -> Self {
        Self::normalize(f64::from(u), 0)
    }
}

impl From<f64> for BigFloat {
    #[inline]
    fn from(f: f64) -> Self {
        Self::normalize(f, 0)
    }
}

impl PartialOrd for BigFloat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        debug_assert!(self.is_normalized());
        debug_assert!(other.is_normalized());
        match self.exponent.cmp(&other.exponent) {
            Ordering::Equal => self.factor.partial_cmp(&other.factor),
            _ if self.is_zero() || other.is_zero() => self.factor.partial_cmp(&other.factor),
            ordering => Some(ordering),
        }
    }
}

impl Number for BigFloat {
    #[inline]
    fn from_ratio(numerator: u64, denominator: u64) -> Self {
        Self::from((numerator as f64) / (denominator as f64))
    }

    /// Exponentiates a `BigFloat`
    ///
    /// ```
    /// use tool::number::*;
    /// use num_traits::{One, Zero};
    /// assert_eq!(BigFloat::zero().exp(), BigFloat::one());
    /// assert_eq!(BigFloat::one().exp(), BigFloat::from(1.0_f64.exp()));
    /// assert_eq!((-BigFloat::one()).exp(), BigFloat::from((-1.0_f64).exp()));
    /// assert_eq!((BigFloat::from(2.0)).exp(), BigFloat::from(2.0_f64.exp()));
    /// assert_eq!((BigFloat::from(-2.0)).exp(), BigFloat::from((-2.0_f64).exp()));
    /// ```
    #[inline]
    fn exp(&self) -> Self {
        let exponent =
            self.factor * f64::powi(2.0, self.exponent as i32) * std::f64::consts::LOG2_E;
        let integer_exponent = exponent as i64;
        let correction = 2.0_f64.powf(exponent - (integer_exponent as f64));
        Self::normalize(correction, integer_exponent)
    }

    /// Takes the natural logarithm of a `BigFloat`
    /// ```
    /// use tool::number::*;
    /// use num_traits::{One, Zero};
    /// assert_eq!(BigFloat::one().log(), BigFloat::zero());
    /// assert_eq!(BigFloat::from(1.0_f64.exp()).log(), BigFloat::from(1.0_f64.exp().ln()));
    /// ```
    #[inline]
    fn log(&self) -> Self {
        let log2 = self.factor.log2() + self.exponent as f64;
        let ln = log2 * std::f64::consts::LN_2;
        Self::from(ln)
    }

    #[inline]
    fn pow(&self, exp: u32) -> Self {
        Self::normalize(
            self.factor.powi(exp.try_into().unwrap()),
            self.exponent * i64::from(exp),
        )
    }

    #[inline]
    fn min(&self, other: &Self) -> Self {
        if self < other {
            *self
        } else {
            *other
        }
    }

    #[inline]
    fn max(&self, other: &Self) -> Self {
        if self > other {
            *self
        } else {
            *other
        }
    }

    #[inline]
    fn abs(&self) -> Self {
        Self {
            factor: self.factor.abs(),
            exponent: self.exponent,
        }
    }
}

impl FloatNumber for BigFloat {
    #[inline]
    fn sqrt(&self) -> Self {
        let exponent = self.exponent.div_euclid(2);
        let factor = if self.exponent.rem_euclid(2) == 0 {
            self.factor.sqrt()
        } else {
            (self.factor * 2.0).sqrt()
        };
        Self::normalize(factor, exponent)
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.factor.is_finite()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        self.factor.is_nan()
    }

    #[inline]
    fn is_infinite(&self) -> bool {
        self.factor.is_infinite()
    }

    #[inline]
    fn nan() -> Self {
        Self::from(f64::NAN)
    }

    #[inline]
    fn infinity() -> Self {
        Self::from(f64::INFINITY)
    }
}

impl IntervalNumber for BigFloat {
    fn next_up(&self) -> Self {
        Self::normalize(F64::from(self.factor).next_up().into(), self.exponent)
    }

    fn next_down(&self) -> Self {
        Self::normalize(F64::from(self.factor).next_down().into(), self.exponent)
    }
}

impl Add for BigFloat {
    type Output = Self;

    #[inline]
    fn add(self, rhs: BigFloat) -> Self::Output {
        let (bigger, smaller) = if self.exponent >= rhs.exponent {
            (self, rhs)
        } else {
            (rhs, self)
        };
        let diff = smaller.exponent - bigger.exponent;
        let factor = bigger.factor + smaller.factor * f64::powi(2.0, diff as i32);
        Self::normalize(factor, bigger.exponent)
    }
}

impl AddAssign for BigFloat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for BigFloat {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: BigFloat) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for BigFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for BigFloat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: BigFloat) -> Self::Output {
        Self::normalize(self.factor * rhs.factor, self.exponent + rhs.exponent)
    }
}

impl MulAssign for BigFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for BigFloat {
    type Output = Self;

    #[inline]
    fn div(self, rhs: BigFloat) -> Self::Output {
        Self::normalize(self.factor / rhs.factor, self.exponent - rhs.exponent)
    }
}

impl DivAssign for BigFloat {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for BigFloat {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            factor: -self.factor,
            ..self
        }
    }
}

impl Display for BigFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", ryu::Buffer::new().format(self.to_f64()))
    }
}
