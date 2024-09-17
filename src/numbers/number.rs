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
        let (a, b) = $crate::numbers::find_distant($a, $b);
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
        let (a, b) = $crate::numbers::find_distant_with($a, $b, $relative_tolerance, $absolute_tolerance);
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
