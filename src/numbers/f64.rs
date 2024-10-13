use std::{
    fmt::{Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::{One, Zero};

use crate::numbers::{FloatNumber, IntervalNumber, Number};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct F64(f64);

impl F64 {
    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        self.0
    }
}

impl From<u64> for F64 {
    #[inline]
    fn from(u: u64) -> Self {
        Self(u as f64)
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
        Self(self.0.abs())
    }
}

impl FloatNumber for F64 {
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
