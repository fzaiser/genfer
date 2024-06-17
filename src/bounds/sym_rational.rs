use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::{One, Zero};

use crate::bounds::linear::LinearExpr;

use super::{
    float_rat::FloatRat,
    sym_poly::{PolyConstraint, SparsePoly},
};

#[derive(Debug, Clone)]
pub struct RationalFunction<T> {
    pub numer: SparsePoly<T>,
    pub denom: SparsePoly<T>,
}

impl<T> PartialEq for RationalFunction<T>
where
    T: Clone + Zero + One + PartialEq + AddAssign + MulAssign + Div<Output = T>,
{
    fn eq(&self, other: &Self) -> bool {
        if self.numer == other.numer && self.denom == other.denom {
            return true;
        }
        self.numer.clone() * other.denom.clone() == other.numer.clone() * self.denom.clone()
    }
}

impl<T> RationalFunction<T>
where
    T: Clone + Zero + One + PartialEq + AddAssign,
{
    pub fn constant(value: T) -> Self {
        Self {
            numer: SparsePoly::constant(value),
            denom: SparsePoly::one(),
        }
    }

    pub fn var(i: usize) -> Self {
        Self {
            numer: SparsePoly::var(i),
            denom: SparsePoly::one(),
        }
    }

    pub fn inverse(self) -> Self {
        Self {
            numer: self.denom,
            denom: self.numer,
        }
    }

    pub fn pow(self, n: i32) -> Self
    where
        T: MulAssign + Div<Output = T>,
        Self: Zero + One,
    {
        if n == 0 {
            Self::one()
        } else if n == 1 || (n >= 0 && self.is_zero()) || self.is_one() {
            self
        } else {
            let numer = self.numer.pow(n.unsigned_abs());
            let denom = self.denom.pow(n.unsigned_abs());
            if n < 0 {
                Self {
                    numer: denom,
                    denom: numer,
                }
            } else {
                Self { numer, denom }
            }
        }
    }

    /// Must equal `rhs`.
    pub fn must_eq(self, rhs: Self) -> PolyConstraint<T> {
        PolyConstraint::Eq(self.numer * rhs.denom, rhs.numer * self.denom)
    }

    /// Must be less than `rhs`.
    pub fn must_lt(self, rhs: Self) -> PolyConstraint<T> {
        PolyConstraint::Lt(self.numer * rhs.denom, rhs.numer * self.denom)
    }

    /// Must be less than or equal to `rhs`.
    pub fn must_le(self, rhs: Self) -> PolyConstraint<T> {
        PolyConstraint::Le(self.numer * rhs.denom, rhs.numer * self.denom)
    }

    /// Must be greater than `rhs`.
    pub fn must_gt(self, rhs: Self) -> PolyConstraint<T> {
        PolyConstraint::Lt(rhs.numer * self.denom, self.numer * rhs.denom)
    }

    /// Must be greater than or equal to `rhs`.
    pub fn must_ge(self, rhs: Self) -> PolyConstraint<T> {
        PolyConstraint::Le(rhs.numer * self.denom, self.numer * rhs.denom)
    }

    pub fn extract_constant(&self) -> Option<T>
    where
        T: Div<Output = T>,
    {
        if let Some(numer) = self.numer.extract_constant() {
            if let Some(denom) = self.denom.extract_constant() {
                return Some(numer.clone() / denom.clone());
            }
        }
        None
    }

    pub fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &T) -> z3::ast::Real<'a>,
    ) -> z3::ast::Real<'a> {
        let numer = self.numer.to_z3(ctx, conv);
        let denom = self.denom.to_z3(ctx, conv);
        numer / denom
    }

    pub fn to_python(&self) -> String
    where
        T: Display,
    {
        format!("({})/({})", self.numer.to_python(), self.denom.to_python())
    }

    pub fn to_python_z3(&self) -> String
    where
        T: Display,
    {
        format!(
            "({})/({})",
            self.numer.to_python_z3(),
            self.denom.to_python_z3()
        )
    }

    pub fn eval(&self, values: &[T]) -> T
    where
        T: MulAssign,
        T: Div<Output = T>,
    {
        self.numer.eval(values) / self.denom.eval(values)
    }

    pub fn derive(&self, var: usize) -> Self
    where
        T: From<u32>
            + Clone
            + Zero
            + One
            + PartialEq
            + Neg<Output = T>
            + AddAssign
            + SubAssign
            + MulAssign,
    {
        let numer = self.numer.derive(var) * self.denom.clone()
            - self.numer.clone() * self.denom.derive(var);
        let denom = self.denom.clone() * self.denom.clone();
        Self { numer, denom }
    }

    pub fn gradient(&self, vars: &[usize]) -> Vec<Self>
    where
        T: From<u32>
            + Clone
            + Zero
            + One
            + PartialEq
            + Neg<Output = T>
            + AddAssign
            + SubAssign
            + MulAssign,
    {
        vars.iter().map(|&var| self.derive(var)).collect()
    }
}

impl RationalFunction<FloatRat> {
    pub fn extract_linear(&self) -> Option<LinearExpr> {
        if let Some(numer) = self.numer.extract_linear() {
            if let Some(denom) = self.denom.extract_constant() {
                return Some(numer / denom.clone());
            }
        }
        None
    }
}

impl<T: From<f64>> From<f64> for RationalFunction<T>
where
    T: From<f64> + Clone + Zero + One + PartialEq + AddAssign,
{
    fn from(value: f64) -> Self {
        Self {
            numer: SparsePoly::constant(value.into()),
            denom: SparsePoly::one(),
        }
    }
}

impl<T> Zero for RationalFunction<T>
where
    T: Clone + Zero + One + PartialEq + AddAssign + MulAssign + Div<Output = T>,
{
    fn zero() -> Self {
        Self::constant(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.numer.is_zero()
    }
}

impl<T> One for RationalFunction<T>
where
    T: Clone + Zero + AddAssign + MulAssign + One + PartialEq + Div<Output = T>,
{
    fn one() -> Self {
        Self::constant(T::one())
    }

    fn is_one(&self) -> bool {
        self.numer == self.denom
    }
}

impl<T> Neg for RationalFunction<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.numer = -self.numer;
        self
    }
}

impl<T> Add for RationalFunction<T>
where
    Self: AddAssign,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for RationalFunction<T>
where
    T: Clone + Zero + One + PartialEq + AddAssign + MulAssign,
    Self: Zero,
{
    fn add_assign(&mut self, rhs: Self) {
        if self.is_zero() {
            *self = rhs;
        } else if rhs.is_zero() {
            // Do nothing
        } else {
            let numer =
                self.numer.clone() * rhs.denom.clone() + rhs.numer.clone() * self.denom.clone();
            let denom = self.denom.clone() * rhs.denom.clone();
            *self = Self { numer, denom };
        }
    }
}

impl<T> Sub for RationalFunction<T>
where
    Self: SubAssign,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for RationalFunction<T>
where
    Self: Neg<Output = Self> + AddAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self += -rhs;
    }
}

impl<T> Mul for RationalFunction<T>
where
    T: Clone + Zero + One + PartialEq + AddAssign + MulAssign + Mul<Output = T> + Div<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else if self.is_one() {
            rhs
        } else if rhs.is_one() {
            self
        } else {
            let numer = self.numer.clone() * rhs.numer.clone();
            let denom = self.denom.clone() * rhs.denom.clone();
            Self { numer, denom }
        }
    }
}

impl<T> MulAssign for RationalFunction<T>
where
    T: Clone + Zero + One + PartialEq + AddAssign + MulAssign + Mul<Output = T> + Div<Output = T>,
{
    fn mul_assign(&mut self, rhs: Self) {
        if self.is_zero() || rhs.is_one() {
            // no change
        } else if rhs.is_zero() || self.is_one() {
            *self = rhs;
        } else {
            self.numer *= rhs.numer;
            self.denom *= rhs.denom;
        }
    }
}

impl<T> Div for RationalFunction<T>
where
    Self: DivAssign,
{
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<T> DivAssign for RationalFunction<T>
where
    SparsePoly<T>: MulAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.numer *= rhs.denom;
        self.denom *= rhs.numer;
    }
}

impl<T> std::fmt::Display for RationalFunction<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]/[{}]", self.numer, self.denom)
    }
}
