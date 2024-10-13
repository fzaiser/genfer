use std::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter, Result},
    num::FpCategory,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use num_traits::{One, Zero};
use rug::ops::Pow;

use super::{FloatNumber, IntervalNumber, Number, F64};

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
    pub(crate) fn from_int(n: impl Into<rug::Integer>) -> Self {
        Self::Frac(Rc::new(rug::Rational::from(n.into())))
    }

    pub(crate) fn from_frac(n: impl Into<rug::Integer>, d: impl Into<rug::Integer>) -> Self {
        Self::Frac(Rc::new(rug::Rational::from((n.into(), d.into()))))
    }
    pub(crate) fn from_rug_rational(r: rug::Rational) -> Self {
        Self::Frac(Rc::new(r))
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Rational::Frac(r) => r.to_f64(),
            Rational::Special(s) => match s {
                Special::NaR => f64::NAN,
                Special::PosInf => f64::INFINITY,
                Special::NegInf => f64::NEG_INFINITY,
            },
        }
    }

    pub fn to_f64_down(&self) -> f64 {
        let rounded = self.to_f64();
        if &Rational::from(rounded) > self {
            F64::from(rounded).next_down().to_f64()
        } else {
            rounded
        }
    }

    pub fn to_f64_up(&self) -> f64 {
        let rounded = self.to_f64();
        if &Rational::from(rounded) < self {
            F64::from(rounded).next_up().to_f64()
        } else {
            rounded
        }
    }

    pub(crate) fn to_integer_ratio(&self) -> (rug::Integer, rug::Integer) {
        match self {
            Self::Frac(r) => (r.numer().clone(), r.denom().clone()),
            Self::Special(s) => match s {
                Special::NaR => (rug::Integer::new(), rug::Integer::new()),
                Special::PosInf => (rug::Integer::from(1), rug::Integer::new()),
                Special::NegInf => (rug::Integer::from(-1), rug::Integer::new()),
            },
        }
    }

    pub(crate) fn not_a_rational() -> Self {
        Self::Special(Special::NaR)
    }

    pub(crate) fn infinity() -> Self {
        Self::Special(Special::PosInf)
    }

    pub(crate) fn neg_infinity() -> Self {
        Self::Special(Special::NegInf)
    }

    pub fn pow(&self, exp: i32) -> Rational {
        if exp == 0 {
            return Self::one();
        }
        if exp == 1 {
            return self.clone();
        }
        if self.is_zero() && exp < 0 {
            return Self::infinity();
        }
        match self {
            Self::Special(s) => match s {
                Special::NaR => Self::not_a_rational(),
                _ if exp < 0 => Self::zero(),
                Special::PosInf => Self::infinity(),
                Special::NegInf => {
                    if exp % 2 == 0 {
                        Self::infinity()
                    } else {
                        Self::neg_infinity()
                    }
                }
            },
            Self::Frac(r) => Self::Frac(Rc::new(rug::Rational::from(r.as_ref().pow(exp)))),
        }
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::Frac(r) => write!(f, "{}", *r),
            Self::Special(s) => match s {
                Special::NaR => write!(f, "(not a rational)"),
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

impl From<u64> for Rational {
    #[inline]
    fn from(n: u64) -> Self {
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
        match self {
            Self::Frac(r) => Self::Frac(Rc::new(rug::Rational::from(r.as_ref()).abs())),
            Self::Special(s) => match s {
                Special::NaR => Self::not_a_rational(),
                Special::PosInf | Special::NegInf => Self::infinity(),
            },
        }
    }
}

impl FloatNumber for Rational {
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
