use std::{fmt::Display, mem};

use num_traits::{One, Zero};
use TaylorExpansion::*;

use crate::number::Number;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TaylorExpansion<T: Clone> {
    // TODO: wrap the fields in an Rc to make cloning cheaper?
    Constant(T),
    Polynomial { coeffs: Box<[T]> },
}

impl<T: Number> TaylorExpansion<T> {
    pub fn var(x: T, order: usize) -> Self {
        let mut coeffs = vec![T::zero(); order + 1].into_boxed_slice();
        if 1 < coeffs.len() {
            coeffs[1] = T::one();
        }
        coeffs[0] = x;
        Polynomial { coeffs }
    }

    pub fn coeff(&self, order: usize) -> T {
        match self {
            Polynomial { coeffs } => coeffs[order].clone(),
            Constant(x) => {
                if order == 0 {
                    x.clone()
                } else {
                    T::zero()
                }
            }
        }
    }

    pub fn order(&self) -> usize {
        match self {
            Polynomial { coeffs } => coeffs.len(),
            Constant(_) => usize::MAX,
        }
    }

    pub fn derivative(&self, order: usize) -> T {
        match self {
            Polynomial { coeffs } => {
                let one = T::one();
                let factorial = (1..=order).fold(one.clone(), |acc, i| acc * T::from(i as u32));
                factorial * coeffs[order].clone()
            }
            Constant(c) => {
                if order == 0 {
                    c.clone()
                } else {
                    T::zero()
                }
            }
        }
    }

    pub fn from_coefficients(coeffs: impl Into<Box<[T]>>) -> Self {
        Polynomial {
            coeffs: coeffs.into(),
        }
    }

    /// Returns the Taylor expansion of the `n`-the coefficient of this Taylor expansion.
    pub fn taylor_expansion_of_coeff(&self, n: usize) -> Self {
        match self {
            Constant(c) => {
                if n == 0 {
                    Constant(c.exp())
                } else {
                    Self::zero()
                }
            }
            Polynomial { coeffs } => {
                let mut res = coeffs[n..].to_owned().into_boxed_slice();
                let res_len = res.len();
                let mut factor = T::one();
                for k in 1..res_len {
                    factor *= T::from((n + k) as u32) / T::from(k as u32);
                    res[k] *= factor.clone();
                }
                Polynomial { coeffs: res }
            }
        }
    }

    /// Substitutes a Taylor expansion for the variables of this Taylor expansion.
    /// The substitution must have the same order as this Taylor expansion.
    pub fn subst(&self, subst: &Self) -> Self {
        match self {
            Constant(_) => self.clone(),
            Polynomial { coeffs } => {
                match subst {
                    Polynomial {
                        coeffs: subst_coeffs,
                    } => assert_eq!(
                        subst_coeffs.len(),
                        coeffs.len(),
                        "Substitution must have the same order"
                    ),
                    Constant(_) => {}
                }
                // Use Horner's method:
                let mut res = Self::zero();
                for c in coeffs.iter().rev() {
                    res = res * subst.clone() + Constant(c.clone());
                }
                res
            }
        }
    }
}

#[test]
pub fn test_taylor_expansion_of_coeff() {
    use crate::number::F64;
    let x = TaylorExpansion::var(F64::from(2.0), 4);
    let f_x = (x.clone() * x + TaylorExpansion::one()).exp();
    let g_x = f_x.taylor_expansion_of_coeff(2);
    assert_eq!(
        g_x,
        TaylorExpansion::from_coefficients(vec![
            1_335.718_431_923_189_4.into(),
            6_530.179_000_513_37.into(),
            17_067.513_296_796_307.into()
        ])
    );
}

#[test]
pub fn test_subst() {
    use crate::number::F64;
    let x = TaylorExpansion::var(F64::one(), 2);
    let y = TaylorExpansion::var(F64::from(2.0), 2);
    assert_eq!(
        x.subst(&y),
        TaylorExpansion::from_coefficients(vec![3.0.into(), 1.0.into(), 0.0.into()])
    );
    let res = (x.clone() * x).subst(&(y.clone() * y));
    assert_eq!(
        res,
        TaylorExpansion::from_coefficients(vec![25.0.into(), 40.0.into(), 26.0.into()])
    );
}

impl<T: Number> Number for TaylorExpansion<T> {
    fn exp(&self) -> Self {
        match self {
            Polynomial { coeffs } => {
                let order = coeffs.len();
                let zero = T::zero();
                let mut res = vec![zero.clone(); order].into_boxed_slice();
                res[0] = coeffs[0].exp();
                for k in 1..order {
                    let sum = (1..=k).fold(zero.clone(), |sum, j| {
                        sum + res[k - j].clone() * coeffs[j].clone() * T::from(j as u32)
                    });
                    res[k] = sum / T::from(k as u32);
                }
                Polynomial { coeffs: res }
            }
            Constant(c) => Constant(c.exp()),
        }
    }

    fn log(&self) -> Self {
        match self {
            Polynomial { coeffs } => {
                let order = coeffs.len();
                let zero = T::zero();
                let mut res = vec![zero.clone(); order].into_boxed_slice();
                res[0] = coeffs[0].log();
                for k in 1..order {
                    let sum = (1..k).fold(zero.clone(), |sum, j| {
                        sum + coeffs[k - j].clone() * res[j].clone() * T::from(j as u32)
                    });
                    res[k] = (coeffs[k].clone() * T::from(k as u32) - sum)
                        / coeffs[0].clone()
                        / T::from(k as u32);
                }
                Polynomial { coeffs: res }
            }
            Constant(c) => Constant(c.log()),
        }
    }

    /// Binary exponentiation
    fn pow(&self, mut exp: u32) -> Self {
        let mut res: Self = Self::one();
        let mut base = self.clone();
        while exp > 0 {
            if exp & 1 == 1 {
                res *= base.clone();
            }
            base *= base.clone();
            exp >>= 1;
        }
        res
    }
}

#[allow(unused)]
fn fast_nonzero_pow<T: Number>(term: &TaylorExpansion<T>, exp: u32) -> TaylorExpansion<T> {
    match term {
        Constant(c) => Constant(c.pow(exp)),
        Polynomial { coeffs } => {
            let order = coeffs.len();
            let zero = T::zero();
            let mut res = vec![zero.clone(); order].into_boxed_slice();
            res[0] = coeffs[0].pow(exp);
            let u0_inv = (T::one() / coeffs[0].clone()).pow(exp);
            let exp_factor = T::from(exp);
            for k in 1..order {
                let sum1 = (1..=k).fold(zero.clone(), |sum, j| {
                    sum + res[k - j].clone() * coeffs[j].clone() * T::from(j as u32)
                });
                let sum2 = (1..k).fold(zero.clone(), |sum, j| {
                    sum + coeffs[k - j].clone() * res[j].clone() * T::from(j as u32)
                });
                res[k] = u0_inv.clone() * (exp_factor.clone() * sum1 - sum2) / T::from(k as u32);
            }
            Polynomial { coeffs: res }
        }
    }
}

impl<T: Number + Zero> Zero for TaylorExpansion<T> {
    fn zero() -> Self {
        Constant(T::zero())
    }
    fn is_zero(&self) -> bool {
        match self {
            Constant(c) => c.is_zero(),
            Polynomial { .. } => false,
        }
    }
}

impl<T: Number + Zero + One + PartialEq> One for TaylorExpansion<T> {
    fn one() -> Self {
        Constant(T::one())
    }
    fn is_one(&self) -> bool {
        match self {
            Constant(c) => c.is_one(),
            Polynomial { .. } => false,
        }
    }
}

impl<T: Clone + From<u32>> From<u32> for TaylorExpansion<T> {
    fn from(x: u32) -> Self {
        Constant(T::from(x))
    }
}

impl<T: Number> std::ops::Add for TaylorExpansion<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: Number> std::ops::AddAssign for TaylorExpansion<T> {
    fn add_assign(&mut self, rhs: Self) {
        match rhs {
            Constant(rhs) => match self {
                Constant(lhs) => *lhs += rhs,
                Polynomial { coeffs } => coeffs[0] += rhs,
            },
            Polynomial { coeffs: mut ws } => {
                let coeffs = match &*self {
                    Constant(c) => {
                        ws[0] += c.clone();
                        ws
                    }
                    Polynomial { coeffs: us } => {
                        let order = us.len().min(ws.len());
                        for i in 0..order {
                            ws[i] += us[i].clone();
                        }
                        let mut coeffs = Vec::new().into_boxed_slice();
                        mem::swap(&mut ws, &mut coeffs);
                        let mut coeffs = coeffs.into_vec();
                        coeffs.truncate(order);
                        coeffs.into_boxed_slice()
                    }
                };
                mem::swap(self, &mut Polynomial { coeffs });
            }
        }
    }
}

impl<T: Number> std::ops::Neg for TaylorExpansion<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Constant(c) => Constant(-c),
            Polynomial { coeffs } => Polynomial {
                coeffs: coeffs.iter().map(|c| -c.clone()).collect(),
            },
        }
    }
}

impl<T: Number> std::ops::Sub for TaylorExpansion<T> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T: Number> std::ops::SubAssign for TaylorExpansion<T> {
    fn sub_assign(&mut self, rhs: Self) {
        match rhs {
            Constant(rhs) => match self {
                Constant(lhs) => *lhs -= rhs,
                Polynomial { coeffs } => coeffs[0] -= rhs,
            },
            Polynomial { coeffs: mut ws } => {
                let coeffs = match &*self {
                    Constant(c) => {
                        for i in 0..ws.len() {
                            ws[i] = -ws[i].clone();
                        }
                        ws[0] += c.clone();
                        ws
                    }
                    Polynomial { coeffs: us } => {
                        let order = us.len().min(ws.len());
                        for i in 0..order {
                            ws[i] = us[i].clone() - ws[i].clone();
                        }
                        let mut coeffs = Vec::new().into_boxed_slice();
                        mem::swap(&mut ws, &mut coeffs);
                        let mut coeffs = coeffs.into_vec();
                        coeffs.truncate(order);
                        coeffs.into_boxed_slice()
                    }
                };
                mem::swap(self, &mut Polynomial { coeffs });
            }
        }
    }
}

impl<T: Number> std::ops::Mul for TaylorExpansion<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Constant(lhs), Constant(rhs)) => Constant(lhs * rhs),
            (Constant(c), Polynomial { mut coeffs }) | (Polynomial { mut coeffs }, Constant(c)) => {
                for coeff in &mut *coeffs {
                    *coeff *= c.clone();
                }
                Polynomial { coeffs }
            }
            (Polynomial { coeffs: us }, Polynomial { coeffs: ws }) => {
                let order = us.len().min(ws.len());
                let zero = T::zero();
                let mut result = vec![zero.clone(); order].into_boxed_slice();
                for k in 0..order {
                    result[k] = (0..=k).fold(zero.clone(), |sum, j| {
                        sum + us[j].clone() * ws[k - j].clone()
                    });
                }
                Polynomial { coeffs: result }
            }
        }
    }
}

impl<T: Number> std::ops::MulAssign for TaylorExpansion<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: Number> std::ops::Div for TaylorExpansion<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Constant(lhs), Constant(rhs)) => Constant(lhs / rhs),
            (Polynomial { mut coeffs }, Constant(c)) => {
                for coeff in &mut *coeffs {
                    *coeff /= c.clone();
                }
                Polynomial { coeffs }
            }
            (Constant(c), Polynomial { coeffs: ws }) => {
                let order = ws.len();
                let zero = T::zero();
                let mut result = vec![zero.clone(); order].into_boxed_slice();
                let scale = T::one() / ws[0].clone();
                result[0] = c * scale.clone();
                for k in 1..order {
                    result[k] = scale.clone()
                        * (0..k).fold(zero.clone(), |sum, i| {
                            sum - result[i].clone() * ws[k - i].clone()
                        });
                }
                Polynomial { coeffs: result }
            }
            (Polynomial { coeffs: us }, Polynomial { coeffs: ws }) => {
                let order = us.len().min(ws.len());
                let zero = T::zero();
                let mut result = vec![zero.clone(); order].into_boxed_slice();
                let scale = T::one() / ws[0].clone();
                result[0] = scale.clone() * us[0].clone();
                for k in 1..order {
                    result[k] = scale.clone()
                        * (0..k).fold(us[k].clone(), |sum, i| {
                            sum - result[i].clone() * ws[k - i].clone()
                        });
                }
                Polynomial { coeffs: result }
            }
        }
    }
}

impl<T: Number> std::ops::DivAssign for TaylorExpansion<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<T: Number + Display> Display for TaylorExpansion<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant(c) => write!(f, "{c}"),
            Polynomial { coeffs } => {
                let mut first = true;
                for i in 0..coeffs.len() {
                    if coeffs[i].is_zero() {
                        continue;
                    }
                    if first {
                        first = false;
                    } else {
                        write!(f, " + ")?;
                    }
                    if i == 0 {
                        write!(f, "{}", coeffs[i])?;
                    } else if i == 1 {
                        write!(f, "{}ε", coeffs[i])?;
                    } else {
                        write!(f, "{}ε^{}", coeffs[i], i)?;
                    }
                }
                if first {
                    write!(f, "0")?;
                }
                Ok(())
            }
        }
    }
}

#[test]
fn test_taylor_e_x_squared_1() {
    use crate::number::F64;
    let x = TaylorExpansion::<F64>::var(F64::zero(), 9);
    let result = (x.clone() * x - One::one()).exp();
    let coeffs: Vec<F64> = vec![
        0.367_879_441_171_442_33.into(),
        0.0.into(),
        0.367_879_441_171_442_33.into(),
        0.0.into(),
        0.183_939_720_585_721_17.into(),
        0.0.into(),
        0.061_313_240_195_240_39.into(),
        0.0.into(),
        0.015_328_310_048_810_098.into(),
        0.0.into(),
    ];
    assert_eq!(result, TaylorExpansion::from_coefficients(coeffs));
}

#[test]
fn test_division() {
    use crate::number::F64;
    let x = TaylorExpansion::<F64>::var(F64::zero(), 9);
    let result = x.clone() / (x.clone() - One::one());
    let coeffs: Vec<F64> = vec![
        0.0.into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
        (-1.0).into(),
    ];
    assert_eq!(result, TaylorExpansion::from_coefficients(coeffs));

    let coeffs: Vec<F64> = vec![
        0.0.into(),
        1.0.into(),
        (-1.0).into(),
        0.5.into(),
        (-0.166_666_666_666_666_63).into(),
        0.041_666_666_666_666_63.into(),
        (-0.008_333_333_333_333_31).into(),
        0.001_388_888_888_888_877.into(),
        (-0.000_198_412_698_412_693_37).into(),
        0.000_024_801_587_301_585_587.into(),
    ];
    let result = x.clone() / x.exp();
    assert_eq!(result, TaylorExpansion::from_coefficients(coeffs));
}

#[test]
fn test_division_constant() {
    use crate::number::F64;
    let x = TaylorExpansion::<F64>::var(F64::zero(), 9);
    let result = TaylorExpansion::one() / (x.clone() - One::one());
    let coeffs: Vec<F64> = vec![(-1.0).into(); 10];
    assert_eq!(result, TaylorExpansion::from_coefficients(coeffs));

    let coeffs: Vec<F64> = vec![
        1.0.into(),
        (-1.0).into(),
        0.5.into(),
        (-0.166_666_666_666_666_63).into(),
        0.041_666_666_666_666_63.into(),
        (-0.008_333_333_333_333_31).into(),
        0.001_388_888_888_888_877.into(),
        (-0.000_198_412_698_412_693_37).into(),
        0.000_024_801_587_301_585_587.into(),
        (-2.755_731_922_398_079_3e-6).into(),
    ];
    let result = TaylorExpansion::one() / x.exp();
    assert_eq!(result, TaylorExpansion::from_coefficients(coeffs));
}

#[test]
fn test_log() {
    use crate::number::F64;
    let x = TaylorExpansion::<F64>::var(F64::one(), 4);
    let result = x.clone().log();
    let coeffs: Vec<F64> = vec![
        0.0.into(),
        1.0.into(),
        (-0.5).into(),
        0.333_333_333_333_333_3.into(),
        (-0.25).into(),
    ];
    assert_eq!(result, TaylorExpansion::from_coefficients(coeffs));
    assert_eq!(x.clone().exp().log(), x.clone());
    assert_eq!(x.clone().log().exp(), x.clone());
    let coeffs: Vec<F64> = vec![1.0.into(), 2.0.into(), 3.0.into()];
    let e = TaylorExpansion::from_coefficients(coeffs);
    let res: Vec<F64> = vec![0.0.into(), 2.0.into(), 1.0.into()];
    assert_eq!(e.log(), TaylorExpansion::from_coefficients(res));
    assert_eq!(e.log().exp(), e);
}
