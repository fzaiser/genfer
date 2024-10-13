use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{One, Zero};

use crate::numbers::{Number, Rational, F64};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct FloatRat {
    rat: Rational,
    float: f64,
}

impl FloatRat {
    pub(crate) fn new(rat: Rational) -> Self {
        let float = rat.to_f64();
        FloatRat { rat, float }
    }

    pub(crate) fn float(&self) -> f64 {
        self.float
    }

    pub fn rat(&self) -> Rational {
        self.rat.clone()
    }

    pub(crate) fn pow(&self, exp: i32) -> Self {
        let rat = self.rat.pow(exp);
        FloatRat::new(rat)
    }
}

impl From<u64> for FloatRat {
    fn from(int: u64) -> Self {
        FloatRat::new(Rational::from(int))
    }
}

impl From<Rational> for FloatRat {
    fn from(rat: Rational) -> Self {
        FloatRat::new(rat)
    }
}

impl From<FloatRat> for Rational {
    fn from(float_rat: FloatRat) -> Rational {
        float_rat.rat()
    }
}

impl Zero for FloatRat {
    fn zero() -> FloatRat {
        FloatRat::new(Rational::zero())
    }

    fn is_zero(&self) -> bool {
        self.rat.is_zero()
    }
}

impl One for FloatRat {
    fn one() -> FloatRat {
        FloatRat::new(Rational::one())
    }

    fn is_one(&self) -> bool {
        self.rat.is_one()
    }
}

impl Neg for FloatRat {
    type Output = FloatRat;

    fn neg(self) -> FloatRat {
        let rat = -self.rat;
        FloatRat::new(rat)
    }
}

impl AddAssign<FloatRat> for FloatRat {
    fn add_assign(&mut self, other: FloatRat) {
        self.rat += other.rat;
        self.float = self.rat.to_f64();
    }
}

impl Add<FloatRat> for FloatRat {
    type Output = FloatRat;

    fn add(self, other: FloatRat) -> FloatRat {
        let rat = self.rat + other.rat;
        FloatRat::new(rat)
    }
}

impl SubAssign<FloatRat> for FloatRat {
    fn sub_assign(&mut self, other: FloatRat) {
        self.rat -= other.rat;
        self.float = self.rat.to_f64();
    }
}

impl Sub<FloatRat> for FloatRat {
    type Output = FloatRat;

    fn sub(self, other: FloatRat) -> FloatRat {
        let rat = self.rat - other.rat;
        FloatRat::new(rat)
    }
}

impl MulAssign<FloatRat> for FloatRat {
    fn mul_assign(&mut self, other: FloatRat) {
        self.rat *= other.rat;
        self.float = self.rat.to_f64();
    }
}

impl Mul<FloatRat> for FloatRat {
    type Output = FloatRat;

    fn mul(self, other: FloatRat) -> FloatRat {
        let rat = self.rat * other.rat;
        FloatRat::new(rat)
    }
}

impl DivAssign<FloatRat> for FloatRat {
    fn div_assign(&mut self, other: FloatRat) {
        self.rat /= other.rat;
        self.float = self.rat.to_f64();
    }
}

impl Div<FloatRat> for FloatRat {
    type Output = FloatRat;

    fn div(self, other: FloatRat) -> FloatRat {
        let rat = self.rat / other.rat;
        FloatRat::new(rat)
    }
}

impl std::fmt::Display for FloatRat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", F64::from(self.float()))
    }
}

impl Number for FloatRat {
    fn exp(&self) -> Self {
        Self::new(self.rat.exp())
    }

    fn log(&self) -> Self {
        Self::new(self.rat.log())
    }

    fn pow(&self, exp: u32) -> Self {
        Self::new(Number::pow(&self.rat, exp))
    }

    fn max(&self, other: &Self) -> Self {
        Self::new(self.rat.max(&other.rat))
    }
}
