use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::{One, Zero};

pub trait Number:
    Clone
    + Display
    + Zero
    + One
    + From<u64>
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
        Self::from(numerator) / Self::from(denominator)
    }
    fn exp(&self) -> Self;
    fn log(&self) -> Self;
    fn pow(&self, exp: u32) -> Self;
    fn max(&self, other: &Self) -> Self;
    fn min(&self, other: &Self) -> Self {
        -((-self.clone()).max(&(-other.clone())))
    }
    fn abs(&self) -> Self {
        self.max(&Self::zero())
    }
}

pub trait FloatNumber: Number {
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
}
