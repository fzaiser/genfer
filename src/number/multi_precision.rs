use std::{
    cell::OnceCell,
    fmt::{Debug, Display, Formatter, Result},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use num_traits::{One, Zero};
use rug::{ops::Pow, Float};

use super::number::{FloatNumber, IntervalNumber, Number};

thread_local! {
    pub static PRECISION: OnceCell<u32> = OnceCell::new();
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

    fn abs(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        Self(Rc::new(self.0.as_ref().clone().abs()))
    }
}

impl FloatNumber for MultiPrecFloat {
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
