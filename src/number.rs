use std::{
    cell::OnceCell,
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result},
    num::FpCategory,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use num_traits::{One, Zero};
use rug::{ops::Pow, Float};

thread_local! {
    pub static PRECISION: OnceCell<u32> = OnceCell::new();
}

pub trait Number:
    Clone
    + Display
    + Zero
    + One
    + From<u32>
    + PartialEq
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
{
    fn from_ratio(numerator: u64, denominator: u64) -> Self {
        let two_to_32 = Self::from(u32::MAX) + Self::one();
        let numer =
            Self::from(numerator as u32) + Self::from((numerator >> 32) as u32) * two_to_32.clone();
        let denom =
            Self::from(denominator as u32) + Self::from((denominator >> 32) as u32) * two_to_32;
        numer / denom
    }
    fn exp(&self) -> Self;
    fn log(&self) -> Self;
    fn pow(&self, exp: u32) -> Self;
}

pub trait FloatNumber: Number {
    fn abs(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn is_finite(&self) -> bool;
    fn is_nan(&self) -> bool;
    fn is_infinite(&self) -> bool;
    fn nan() -> Self;
    fn infinity() -> Self;
}

pub trait IntervalNumber: FloatNumber + PartialOrd {
    /// Check whether two numbers are close to each other
    ///
    /// Relative tolerance is with respect to the second number because it is usually the expected value.
    #[inline]
    fn is_close_with(
        &self,
        other: &Self,
        relative_tolerance: &Self,
        absolute_tolerance: &Self,
    ) -> bool {
        let diff = (self.clone() - other.clone()).abs();
        diff <= absolute_tolerance.clone() || diff <= relative_tolerance.clone() * other.abs()
    }
    #[inline]
    fn is_close(&self, other: &Self) -> bool {
        self.is_close_with(
            other,
            &Self::from_ratio(1, 1_000_000_000),
            &Self::from_ratio(1, 100_000_000),
        )
    }
    fn next_up(&self) -> Self;
    fn next_down(&self) -> Self;
    #[inline]
    fn min(&self, other: &Self) -> Self {
        if self < other {
            self.clone()
        } else {
            other.clone()
        }
    }
    #[inline]
    fn max(&self, other: &Self) -> Self {
        if self > other {
            self.clone()
        } else {
            other.clone()
        }
    }
}

#[inline]
pub fn all_close<'a, 'b, T: IntervalNumber + 'a + 'b>(
    a: impl IntoIterator<Item = &'a T>,
    b: impl IntoIterator<Item = &'b T>,
) -> bool {
    matches!(find_distant(a, b), (None, None))
}

#[inline]
pub fn find_distant<'a, 'b, T: IntervalNumber + 'a + 'b>(
    a: impl IntoIterator<Item = &'a T>,
    b: impl IntoIterator<Item = &'b T>,
) -> (Option<&'a T>, Option<&'b T>) {
    let mut a_iter = a.into_iter();
    let mut b_iter = b.into_iter();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a), Some(b)) => {
                if !a.is_close(b) {
                    return (Some(a), Some(b));
                }
            }
            (None, None) => return (None, None),
            (a, b) => return (a, b),
        }
    }
}

#[inline]
pub fn all_close_with<'a, 'b, T: IntervalNumber + 'a + 'b>(
    a: impl IntoIterator<Item = &'a T>,
    b: impl IntoIterator<Item = &'a T>,
    relative_tolerance: &T,
    absolute_tolerance: &T,
) -> bool {
    matches!(
        find_distant_with(a, b, relative_tolerance, absolute_tolerance),
        (None, None)
    )
}

#[inline]
pub fn find_distant_with<'a, 'b, T: IntervalNumber + 'a + 'b>(
    a: impl IntoIterator<Item = &'a T>,
    b: impl IntoIterator<Item = &'b T>,
    relative_tolerance: &T,
    absolute_tolerance: &T,
) -> (Option<&'a T>, Option<&'b T>) {
    let mut a_iter = a.into_iter();
    let mut b_iter = b.into_iter();
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a), Some(b)) => {
                if !a.is_close_with(b, relative_tolerance, absolute_tolerance) {
                    return (Some(a), Some(b));
                }
            }
            (None, None) => return (None, None),
            (a, b) => return (a, b),
        }
    }
}

#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr) => {
        assert!(
            $a.is_close(&*$b),
            "assertion failed: `is_close(left, right)`\nleft:  {}\nright: {}",
            $a,
            $b,
        )
    };
    ($a:expr, $b:expr, $relative_tolerance:expr, $absolute_tolerance:expr) => {
        assert!(
            $a.is_close_with(&*$b, &*$relative_tolerance, &*$absolute_tolerance),
            "assertion failed: `is_close(left, right, relative_tol = {}, absolute_tol = {})`\nleft:  {}\nright: {}",
            $relative_tolerance,
            $absolute_tolerance,
            $a,
            $b,
        )
    };
}

#[macro_export]
macro_rules! assert_all_close {
    ($a:expr, $b:expr $(,)?) => {
        let (a, b) = $crate::number::find_distant($a, $b);
        if let (None, None) = (a, b) {} else {
            panic!(
                "assertion failed: `all_close(left, right)`\nleft:  {}\nright: {}\nThese values differ:\nleft:  {}\nright: {}",
                $a,
                $b,
                a.map_or("None".to_string(), std::string::ToString::to_string),
                b.map_or("None".to_string(), std::string::ToString::to_string),
            )
        }
    };
    ($a:expr, $b:expr, $(rel_tol =)? $relative_tolerance:expr, $(abs_tol =)? $absolute_tolerance:expr $(,)?) => {
        let (a, b) = $crate::number::find_distant_with($a, $b, $relative_tolerance, $absolute_tolerance);
        if let (None, None) = (a, b) {} else {
            panic!(
            "assertion failed: `all_close(left, right, relative_tol = {}, absolute_tol = {})`\nleft:  {}\nright: {}",
            $relative_tolerance,
            $absolute_tolerance,
            $a,
            $b,
                a.map_or("None".to_string(), std::string::ToString::to_string),
                b.map_or("None".to_string(), std::string::ToString::to_string),
            )
        }
    };
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct F64(f64);

impl F64 {
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0
    }
}

impl From<u32> for F64 {
    #[inline]
    fn from(u: u32) -> Self {
        Self(f64::from(u))
    }
}

impl From<f64> for F64 {
    #[inline]
    fn from(f: f64) -> Self {
        Self(f)
    }
}

impl From<F64> for f64 {
    #[inline]
    fn from(f: F64) -> Self {
        f.0
    }
}

impl Display for F64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", ryu::Buffer::new().format(self.0))
    }
}

impl Number for F64 {
    #[inline]
    fn from_ratio(numerator: u64, denominator: u64) -> Self {
        Self((numerator as f64) / (denominator as f64))
    }

    #[inline]
    fn exp(&self) -> Self {
        f64::exp(self.0).into()
    }

    #[inline]
    fn log(&self) -> Self {
        f64::ln(self.0).into()
    }

    #[inline]
    fn pow(&self, exp: u32) -> Self {
        f64::powi(self.0, exp.try_into().unwrap()).into()
    }
}

impl FloatNumber for F64 {
    #[inline]
    fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        Self(self.0.sqrt())
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    #[inline]
    fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }

    #[inline]
    fn nan() -> Self {
        Self(f64::NAN)
    }

    #[inline]
    fn infinity() -> Self {
        Self(f64::INFINITY)
    }
}

impl IntervalNumber for F64 {
    /// Copied from the unstable function in the  Rust standard library.
    #[inline]
    fn next_up(&self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const TINY_BITS: u64 = 0x1; // Smallest positive f64.
        const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

        let bits = self.0.to_bits();
        if self.0.is_nan() || bits == f64::INFINITY.to_bits() {
            return *self;
        }

        let abs = bits & CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            TINY_BITS
        } else if bits == abs {
            bits + 1
        } else {
            bits - 1
        };
        Self(f64::from_bits(next_bits))
    }

    /// Copied from the unstable function in the  Rust standard library.
    #[inline]
    fn next_down(&self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const NEG_TINY_BITS: u64 = 0x8000_0000_0000_0001; // Smallest (in magnitude) negative f64.
        const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

        let bits = self.0.to_bits();
        if self.0.is_nan() || bits == f64::NEG_INFINITY.to_bits() {
            return *self;
        }

        let abs = bits & CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            NEG_TINY_BITS
        } else if bits == abs {
            bits - 1
        } else {
            bits + 1
        };
        Self(f64::from_bits(next_bits))
    }
}

impl Zero for F64 {
    #[inline]
    fn zero() -> Self {
        Self(0.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for F64 {
    #[inline]
    fn one() -> Self {
        Self(1.0)
    }
}

impl Neg for F64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Add for F64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for F64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for F64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for F64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for F64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign for F64 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

/// Extracts the exponent and sets it to zero
///
/// ```
/// use genfer::number::extract_exponent;
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
    /// use genfer::number::*;
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
    /// use genfer::number::*;
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
}

impl FloatNumber for BigFloat {
    #[inline]
    fn abs(&self) -> Self {
        Self {
            factor: self.factor.abs(),
            exponent: self.exponent,
        }
    }

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

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiPrecFloat(Rc<rug::Float>);

impl From<u32> for MultiPrecFloat {
    #[inline]
    fn from(u: u32) -> Self {
        Self(Rc::new(Float::with_val(
            PRECISION.with(|p| *p.get().unwrap_or(&0)),
            u,
        )))
    }
}

impl From<f64> for MultiPrecFloat {
    #[inline]
    fn from(f: f64) -> Self {
        Self(Rc::new(Float::with_val(
            PRECISION.with(|p| *p.get().unwrap_or(&0)),
            f,
        )))
    }
}

impl From<MultiPrecFloat> for f64 {
    #[inline]
    fn from(f: MultiPrecFloat) -> Self {
        f.0.to_f64()
    }
}

impl Display for MultiPrecFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", *self.0)
    }
}

impl Zero for MultiPrecFloat {
    #[inline]
    fn zero() -> Self {
        Self::from(0.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for MultiPrecFloat {
    #[inline]
    fn one() -> Self {
        Self::from(1.0)
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self.0 == 1.0
    }
}

impl Neg for MultiPrecFloat {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            return self;
        }
        Self(Rc::new(rug::Float::with_val(self.0.prec(), -&*self.0)))
    }
}

impl Add for MultiPrecFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        Self(Rc::new(rug::Float::with_val(
            self.0.prec(),
            &*self.0 + &*rhs.0,
        )))
    }
}

impl AddAssign for MultiPrecFloat {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if rhs.is_zero() {
            return;
        }
        *self = self.clone() + rhs;
    }
}

impl Sub for MultiPrecFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if self.is_zero() {
            return -rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        Self(Rc::new(rug::Float::with_val(
            self.0.prec(),
            &*self.0 - &*rhs.0,
        )))
    }
}

impl SubAssign for MultiPrecFloat {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if rhs.is_zero() {
            return;
        }
        *self = self.clone() - rhs;
    }
}

impl Mul for MultiPrecFloat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if self.is_zero() {
            return self;
        } else if rhs.is_zero() {
            return rhs;
        }
        Self(Rc::new(rug::Float::with_val(
            self.0.prec(),
            &*self.0 * &*rhs.0,
        )))
    }
}

impl MulAssign for MultiPrecFloat {
    fn mul_assign(&mut self, mut rhs: Self) {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if self.is_zero() || rhs.is_one() {
            // nothing to do
        } else if self.is_one() || rhs.is_zero() {
            std::mem::swap(self, &mut rhs);
        } else {
            *self = self.clone() * rhs;
        }
    }
}

impl Div for MultiPrecFloat {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if rhs.is_zero() {
            panic!("Division by zero")
        } else if self.is_zero() || rhs.is_one() {
            return self;
        }
        Self(Rc::new(rug::Float::with_val(
            self.0.prec(),
            &*self.0 / &*rhs.0,
        )))
    }
}

impl DivAssign for MultiPrecFloat {
    fn div_assign(&mut self, rhs: Self) {
        assert_eq!(self.0.prec(), rhs.0.prec());
        if rhs.is_zero() {
            panic!("Division by zero")
        } else if self.is_zero() || rhs.is_one() {
            return;
        }
        *self = self.clone() / rhs;
    }
}

impl Number for MultiPrecFloat {
    fn from_ratio(numerator: u64, denominator: u64) -> Self {
        Self(Rc::new(rug::Float::with_val(
            PRECISION.with(|p| *p.get().unwrap_or(&0)),
            rug::Rational::from((numerator, denominator)),
        )))
    }
    fn exp(&self) -> Self {
        Self(Rc::new(self.0.as_ref().clone().exp()))
    }

    fn log(&self) -> Self {
        Self(Rc::new(self.0.as_ref().clone().ln()))
    }

    fn pow(&self, exp: u32) -> Self {
        Self(Rc::new(self.0.as_ref().clone().pow(exp)))
    }
}

impl FloatNumber for MultiPrecFloat {
    fn abs(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        Self(Rc::new(self.0.as_ref().clone().abs()))
    }

    fn sqrt(&self) -> Self {
        if self.is_zero() || self.is_one() {
            return self.clone();
        }
        Self(Rc::new(self.0.as_ref().clone().sqrt()))
    }

    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }

    fn nan() -> Self {
        Self::from(f64::NAN)
    }

    fn infinity() -> Self {
        Self::from(f64::INFINITY)
    }
}

impl IntervalNumber for MultiPrecFloat {
    fn next_up(&self) -> Self {
        let mut next_up = self.0.as_ref().clone();
        next_up.next_up();
        Self(Rc::new(next_up))
    }

    fn next_down(&self) -> Self {
        let mut next_down = self.0.as_ref().clone();
        next_down.next_down();
        Self(Rc::new(next_down))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Special {
    NaR,
    PosInf,
    NegInf,
}

impl PartialEq for Special {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::PosInf, Self::PosInf) | (Self::NegInf, Self::NegInf)
        )
    }
}

impl PartialOrd for Special {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::PosInf, Self::PosInf) | (Self::NegInf, Self::NegInf) => Some(Ordering::Equal),
            (Self::NegInf, Self::PosInf) => Some(Ordering::Less),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Rational {
    Frac(Rc<rug::Rational>),
    Special(Special),
}

impl Rational {
    pub fn from_int(n: impl Into<rug::Integer>) -> Self {
        Self::Frac(Rc::new(rug::Rational::from(n.into())))
    }

    pub fn from_frac(n: impl Into<rug::Integer>, d: impl Into<rug::Integer>) -> Self {
        Self::Frac(Rc::new(rug::Rational::from((n.into(), d.into()))))
    }
    pub(crate) fn from_rug_rational(r: rug::Rational) -> Self {
        Self::Frac(Rc::new(r))
    }

    pub fn not_a_rational() -> Self {
        Self::Special(Special::NaR)
    }

    pub fn infinity() -> Self {
        Self::Special(Special::PosInf)
    }

    pub fn neg_infinity() -> Self {
        Self::Special(Special::NegInf)
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Frac(r) => write!(f, "{}", *r),
            Self::Special(s) => match s {
                Special::NaR => write!(f, "NaN"),
                Special::PosInf => write!(f, "∞"),
                Special::NegInf => write!(f, "-∞"),
            },
        }
    }
}

impl Zero for Rational {
    #[inline]
    fn zero() -> Self {
        Self::Frac(Rc::new(rug::Rational::new()))
    }

    #[inline]
    fn is_zero(&self) -> bool {
        match self {
            Self::Frac(r) => r.cmp0() == Ordering::Equal,
            Self::Special(_) => false,
        }
    }
}

impl One for Rational {
    #[inline]
    fn one() -> Self {
        Self::Frac(Rc::new(rug::Rational::from(1)))
    }

    #[inline]
    fn is_one(&self) -> bool {
        match self {
            Self::Frac(r) => r.as_ref() == &1,
            Self::Special(_) => false,
        }
    }
}

impl From<u32> for Rational {
    #[inline]
    fn from(n: u32) -> Self {
        Self::from_int(n)
    }
}

impl From<f64> for Rational {
    #[inline]
    fn from(f: f64) -> Self {
        match f.classify() {
            FpCategory::Nan => Self::not_a_rational(),
            FpCategory::Infinite => {
                if f.is_sign_negative() {
                    Self::neg_infinity()
                } else {
                    Self::infinity()
                }
            }
            _ => Self::from_rug_rational(rug::Rational::from_f64(f).unwrap()),
        }
    }
}

impl From<Rational> for f64 {
    #[inline]
    fn from(r: Rational) -> Self {
        match r {
            Rational::Frac(r) => r.to_f64(),
            Rational::Special(s) => match s {
                Special::NaR => f64::NAN,
                Special::PosInf => f64::INFINITY,
                Special::NegInf => f64::NEG_INFINITY,
            },
        }
    }
}

impl Neg for Rational {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Self::Frac(r) => Self::Frac(Rc::new(rug::Rational::from(-&*r))),
            Self::Special(s) => match s {
                Special::NaR => Self::not_a_rational(),
                Special::PosInf => Self::neg_infinity(),
                Special::NegInf => Self::infinity(),
            },
        }
    }
}

impl Add for Rational {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Frac(l), Self::Frac(r)) => Self::Frac(Rc::new(rug::Rational::from(&*l + &*r))),
            (Self::Special(Special::NaR), _)
            | (_, Self::Special(Special::NaR))
            | (Self::Special(Special::PosInf), Self::Special(Special::NegInf))
            | (Self::Special(Special::NegInf), Self::Special(Special::PosInf)) => {
                Self::not_a_rational()
            }
            (Self::Special(s), _) | (_, Self::Special(s)) => Self::Special(s),
        }
    }
}

impl AddAssign for Rational {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Sub for Rational {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Frac(l), Self::Frac(r)) => Self::Frac(Rc::new(rug::Rational::from(&*l - &*r))),
            (Self::Special(Special::NaR), _)
            | (_, Self::Special(Special::NaR))
            | (Self::Special(Special::PosInf), Self::Special(Special::PosInf))
            | (Self::Special(Special::NegInf), Self::Special(Special::NegInf)) => {
                Self::not_a_rational()
            }
            (Self::Special(Special::PosInf), _) | (_, Self::Special(Special::NegInf)) => {
                Self::Special(Special::PosInf)
            }
            (Self::Special(Special::NegInf), _) | (_, Self::Special(Special::PosInf)) => {
                Self::Special(Special::NegInf)
            }
        }
    }
}

impl SubAssign for Rational {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl Mul for Rational {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Frac(l), Self::Frac(r)) => Self::Frac(Rc::new(rug::Rational::from(&*l * &*r))),
            (Self::Special(Special::NaR), _) | (_, Self::Special(Special::NaR)) => {
                Self::not_a_rational()
            }
            (Self::Special(Special::PosInf), Self::Special(Special::PosInf))
            | (Self::Special(Special::NegInf), Self::Special(Special::NegInf)) => Self::infinity(),
            (Self::Special(Special::PosInf), Self::Special(Special::NegInf))
            | (Self::Special(Special::NegInf), Self::Special(Special::PosInf)) => {
                Self::neg_infinity()
            }
            (Self::Special(Special::PosInf), Self::Frac(other))
            | (Self::Frac(other), Self::Special(Special::PosInf)) => match other.cmp0() {
                Ordering::Equal => Self::not_a_rational(),
                Ordering::Greater => Self::infinity(),
                Ordering::Less => Self::neg_infinity(),
            },
            (Self::Special(Special::NegInf), Self::Frac(other))
            | (Self::Frac(other), Self::Special(Special::NegInf)) => match other.cmp0() {
                Ordering::Equal => Self::not_a_rational(),
                Ordering::Greater => Self::neg_infinity(),
                Ordering::Less => Self::infinity(),
            },
        }
    }
}

impl MulAssign for Rational {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl Div for Rational {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Frac(l), Self::Frac(r)) => {
                if r.cmp0() == Ordering::Equal {
                    match l.cmp0() {
                        Ordering::Equal => Self::not_a_rational(),
                        Ordering::Greater => Self::infinity(),
                        Ordering::Less => Self::neg_infinity(),
                    }
                } else {
                    Self::Frac(Rc::new(rug::Rational::from(&*l / &*r)))
                }
            }
            (Self::Special(Special::NaR), _)
            | (_, Self::Special(Special::NaR))
            | (Self::Special(_), Self::Special(_)) => Self::not_a_rational(),
            (Self::Frac(_), Self::Special(Special::PosInf | Special::NegInf)) => Self::zero(),
            (Self::Special(Special::PosInf), Self::Frac(other)) => match other.cmp0() {
                Ordering::Equal | Ordering::Greater => Self::infinity(),
                Ordering::Less => Self::neg_infinity(),
            },
            (Self::Special(Special::NegInf), Self::Frac(other)) => match other.cmp0() {
                Ordering::Equal | Ordering::Greater => Self::neg_infinity(),
                Ordering::Less => Self::infinity(),
            },
        }
    }
}

impl DivAssign for Rational {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Frac(l), Self::Frac(r)) => l.partial_cmp(r),
            (Self::Special(l), Self::Special(r)) => l.partial_cmp(r),
            (Self::Frac(_), Self::Special(Special::PosInf))
            | (Self::Special(Special::NegInf), Self::Frac(_)) => Some(Ordering::Less),
            (Self::Frac(_), Self::Special(Special::NegInf))
            | (Self::Special(Special::PosInf), Self::Frac(_)) => Some(Ordering::Greater),
            _ => None,
        }
    }
}

impl Number for Rational {
    fn from_ratio(numerator: u64, denominator: u64) -> Self {
        Self::from_frac(numerator, denominator)
    }

    fn exp(&self) -> Self {
        match self {
            Self::Special(Special::NaR) => Self::not_a_rational(),
            Self::Special(Special::NegInf) => Self::zero(),
            Self::Special(Special::PosInf) => Self::infinity(),
            _ if self.is_zero() => Self::one(),
            _ => Self::not_a_rational(),
        }
    }

    fn log(&self) -> Self {
        match self {
            Self::Special(Special::NaR | Special::NegInf) => Self::not_a_rational(),
            Self::Special(Special::PosInf) => Self::infinity(),
            _ if self.is_zero() => Self::neg_infinity(),
            _ if self.is_one() => Self::zero(),
            _ => Self::not_a_rational(),
        }
    }

    fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::one();
        }
        if exp == 1 {
            return self.clone();
        }
        match self {
            Self::Special(Special::NaR) => Self::not_a_rational(),
            Self::Special(Special::NegInf) => {
                if exp % 2 == 0 {
                    Self::infinity()
                } else {
                    Self::neg_infinity()
                }
            }
            Self::Special(Special::PosInf) => Self::infinity(),
            Self::Frac(r) => Self::Frac(Rc::new(rug::Rational::from(r.as_ref().pow(exp)))),
        }
    }
}

impl FloatNumber for Rational {
    fn abs(&self) -> Self {
        match self {
            Self::Frac(r) => Self::Frac(Rc::new(rug::Rational::from(r.as_ref()).abs())),
            Self::Special(s) => match s {
                Special::NaR => Self::not_a_rational(),
                Special::PosInf | Special::NegInf => Self::infinity(),
            },
        }
    }

    fn sqrt(&self) -> Self {
        match self {
            Self::Frac(r) => match r.cmp0() {
                Ordering::Equal => Self::zero(),
                Ordering::Greater => {
                    let denom = r.denom().clone();
                    let (denom_sqrt, denom_rem) = denom.sqrt_rem(rug::Integer::new());
                    let numer = r.numer().clone();
                    let (numer_sqrt, numer_rem) = numer.sqrt_rem(rug::Integer::new());
                    if denom_rem.cmp0() == Ordering::Equal && numer_rem.cmp0() == Ordering::Equal {
                        Self::Frac(Rc::new(rug::Rational::from((numer_sqrt, denom_sqrt))))
                    } else {
                        Self::not_a_rational()
                    }
                }
                Ordering::Less => Self::not_a_rational(),
            },
            Self::Special(s) => match s {
                Special::NaR | Special::NegInf => Self::not_a_rational(),
                Special::PosInf => Self::infinity(),
            },
        }
    }

    fn is_finite(&self) -> bool {
        matches!(self, Self::Frac(_))
    }

    fn is_nan(&self) -> bool {
        matches!(self, Self::Special(Special::NaR))
    }

    fn is_infinite(&self) -> bool {
        matches!(self, Self::Special(Special::PosInf | Special::NegInf))
    }

    fn nan() -> Self {
        Self::not_a_rational()
    }

    fn infinity() -> Self {
        Self::infinity()
    }
}

impl IntervalNumber for Rational {
    fn next_up(&self) -> Self {
        self.clone()
    }

    fn next_down(&self) -> Self {
        self.clone()
    }
}
