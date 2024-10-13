use std::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, RangeInclusive, Sub, SubAssign},
};

use num_traits::{One, Zero};

use crate::numbers::{FloatNumber, IntervalNumber, Number};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval<T> {
    pub lo: T,
    pub hi: T,
}

impl<T: IntervalNumber> Interval<T> {
    #[inline]
    pub fn exact(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }

    #[inline]
    pub fn precisely(x: T) -> Self {
        Self::exact(x.clone(), x)
    }

    #[inline]
    pub(crate) fn widen(lo: T, hi: T) -> Self {
        Self::exact(lo.next_down(), hi.next_up())
    }

    #[inline]
    pub(crate) fn contains(&self, x: &T) -> bool {
        self.lo <= *x && *x <= self.hi
    }

    #[inline]
    pub(crate) fn union(&self, x: &T) -> Self {
        Self::exact(self.lo.min(x), self.hi.max(x))
    }

    #[inline]
    pub fn all_reals() -> Self {
        Self::exact(-T::infinity(), T::infinity())
    }
}

impl<T: IntervalNumber> From<u64> for Interval<T> {
    #[inline]
    fn from(value: u64) -> Self {
        Self::exact(T::from(value), T::from(value))
    }
}

impl<T: IntervalNumber> From<RangeInclusive<T>> for Interval<T> {
    #[inline]
    fn from(range: RangeInclusive<T>) -> Self {
        Self::exact(range.start().clone(), range.end().clone())
    }
}

impl<T: IntervalNumber> Zero for Interval<T> {
    #[inline]
    fn zero() -> Self {
        Self::exact(T::zero(), T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.lo.is_zero() && self.hi.is_zero()
    }
}

impl<T: IntervalNumber> One for Interval<T> {
    #[inline]
    fn one() -> Self {
        Self::exact(T::one(), T::one())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.lo.is_one() && self.hi.is_one()
    }
}

impl<T: IntervalNumber> Neg for Interval<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::exact(-self.hi, -self.lo)
    }
}

impl<T: IntervalNumber> Add for Interval<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        Self::widen(self.lo + rhs.lo, self.hi + rhs.hi)
    }
}

impl<T: IntervalNumber> AddAssign for Interval<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<T: IntervalNumber> Sub for Interval<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: IntervalNumber> SubAssign for Interval<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<T: IntervalNumber> Mul for Interval<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if (self.is_zero() && rhs.is_finite()) || (self.is_finite() && rhs.is_zero()) {
            return Self::zero();
        }
        if self.is_one() {
            return rhs;
        }
        if rhs.is_one() {
            return self;
        }
        if (-self.clone()).is_one() {
            return -rhs;
        }
        if (-rhs.clone()).is_one() {
            return -self;
        }
        let a = self.lo.clone() * rhs.lo.clone();
        let b = self.lo * rhs.hi.clone();
        let c = self.hi.clone() * rhs.lo;
        let d = self.hi * rhs.hi;
        Self::widen(a.min(&b).min(&c).min(&d), a.max(&b).max(&c).max(&d))
    }
}

impl<T: IntervalNumber> MulAssign for Interval<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: IntervalNumber> Div for Interval<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        if self.is_nan() || rhs.is_nan() {
            return Self::nan();
        }
        if self.is_zero() && !rhs.is_zero() {
            return self;
        }
        if rhs.is_one() {
            return self;
        }
        let (mut lo, mut hi) = (T::infinity(), -T::infinity());
        if rhs.contains(&T::zero()) {
            if T::zero() <= self.lo {
                hi = T::infinity();
            } else {
                lo = -T::infinity();
            }
            if self.hi <= T::zero() {
                lo = -T::infinity();
            } else {
                hi = T::infinity();
            }
        }
        let a = self.lo.clone() / rhs.lo.clone();
        let b = self.lo / rhs.hi.clone();
        let c = self.hi.clone() / rhs.lo;
        let d = self.hi / rhs.hi;
        let lo = lo.min(&a).min(&b).min(&c).min(&d);
        let hi = hi.max(&a).max(&b).max(&c).max(&d);
        Self::widen(lo, hi)
    }
}

impl<T: IntervalNumber> DivAssign for Interval<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<T: IntervalNumber> Display for Interval<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.lo, self.hi)
    }
}

impl<T: IntervalNumber> PartialOrd for Interval<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.lo == other.lo && other.hi == self.hi {
            Some(Ordering::Equal)
        } else if self.hi <= other.lo {
            Some(Ordering::Less)
        } else if self.lo >= other.hi {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl<T: IntervalNumber> Number for Interval<T> {
    fn exp(&self) -> Self {
        if self.is_zero() {
            return Self::one();
        }
        Self::widen(self.lo.exp(), self.hi.exp())
    }

    fn log(&self) -> Self {
        if self.is_one() {
            return Self::zero();
        }
        Self::widen(self.lo.log(), self.hi.log())
    }

    fn pow(&self, exp: u32) -> Self {
        let result = Self::widen(self.lo.pow(exp), self.hi.pow(exp));
        if self.contains(&T::zero()) {
            result.union(&T::zero())
        } else {
            result
        }
    }

    #[inline]
    fn min(&self, other: &Self) -> Self {
        Self::exact(self.lo.min(&other.lo), self.hi.min(&other.hi))
    }

    #[inline]
    fn max(&self, other: &Self) -> Self {
        Self::exact(self.lo.max(&other.lo), self.hi.max(&other.hi))
    }

    fn abs(&self) -> Self {
        let result = Self::widen(self.lo.abs(), self.hi.abs());
        if self.contains(&T::zero()) {
            result.union(&T::zero())
        } else {
            result
        }
    }
}

impl<T: IntervalNumber> FloatNumber for Interval<T> {
    fn sqrt(&self) -> Self {
        let lo = if self.lo < T::zero() {
            T::zero()
        } else {
            self.lo.sqrt()
        };
        Self::widen(lo, self.hi.sqrt())
    }

    fn is_finite(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    fn is_nan(&self) -> bool {
        self.lo.is_nan() || self.hi.is_nan()
    }

    fn is_infinite(&self) -> bool {
        self.lo.is_infinite() || self.hi.is_infinite()
    }

    fn nan() -> Self {
        Self::exact(T::nan(), T::nan())
    }

    fn infinity() -> Self {
        Self::exact(T::infinity(), T::infinity())
    }
}
