use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use ndarray::ArrayView1;
use num_traits::{One, Zero};
use rug::{ops::Pow, Rational};

use crate::bounds::linear::{LinearConstraint, LinearExpr};

use super::{
    sym_poly::PolyConstraint,
    sym_rational::RationalFunction,
    util::{norm, pow},
};

#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr<T> {
    Constant(T),
    Variable(usize),
    Add(Rc<SymExpr<T>>, Rc<SymExpr<T>>),
    Mul(Rc<SymExpr<T>>, Rc<SymExpr<T>>),
    Pow(Rc<SymExpr<T>>, i32),
}

impl<T> SymExpr<T> {
    pub fn var(i: usize) -> Self {
        Self::Variable(i)
    }

    pub fn inverse(self) -> Self
    where
        T: PartialEq,
        Self: Zero + One,
    {
        self.pow(-1)
    }

    pub fn pow(self, n: i32) -> Self
    where
        T: PartialEq,
        Self: Zero + One,
    {
        if n == 0 {
            Self::one()
        } else if n == 1 || (n >= 0 && self.is_zero()) || self.is_one() {
            self
        } else {
            Self::Pow(Rc::new(self), n)
        }
    }

    /// Must equal `rhs`.
    pub fn must_eq(self, rhs: Self) -> SymConstraint<T> {
        SymConstraint::Eq(self, rhs)
    }

    /// Must be less than `rhs`.
    pub fn must_lt(self, rhs: Self) -> SymConstraint<T> {
        SymConstraint::Lt(self, rhs)
    }

    /// Must be less than or equal to `rhs`.
    pub fn must_le(self, rhs: Self) -> SymConstraint<T> {
        SymConstraint::Le(self, rhs)
    }

    /// Must be greater than `rhs`.
    pub fn must_gt(self, rhs: Self) -> SymConstraint<T> {
        SymConstraint::Lt(rhs, self)
    }

    /// Must be greater than or equal to `rhs`.
    pub fn must_ge(self, rhs: Self) -> SymConstraint<T> {
        SymConstraint::Le(rhs, self)
    }

    pub fn substitute(&self, replacements: &[Self]) -> Self
    where
        T: PartialEq,
        Self: Clone + Zero + One + Add<Output = Self> + Mul<Output = Self>,
    {
        match self {
            Self::Constant(_) => self.clone(),
            Self::Variable(i) => replacements[*i].clone(),
            Self::Add(lhs, rhs) => lhs.substitute(replacements) + rhs.substitute(replacements),
            Self::Mul(lhs, rhs) => lhs.substitute(replacements) * rhs.substitute(replacements),
            Self::Pow(base, n) => base.substitute(replacements).pow(*n),
        }
    }

    pub fn extract_constant(&self) -> Option<T>
    where
        T: Clone,
    {
        match self {
            Self::Constant(c) => Some(c.clone()),
            _ => None,
        }
    }

    pub fn extract_linear(&self) -> Option<LinearExpr<T>>
    where
        T: Clone + Zero + One + AddAssign + Add<Output = T> + MulAssign + Div<Output = T>,
    {
        match self {
            Self::Constant(c) => Some(LinearExpr::constant(c.clone())),
            Self::Variable(i) => Some(LinearExpr::var(*i)),
            Self::Add(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                Some(lhs + rhs)
            }
            Self::Mul(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                if let Some(factor) = lhs.as_constant() {
                    Some(rhs * factor.clone())
                } else {
                    rhs.as_constant().map(|factor| lhs * factor.clone())
                }
            }
            Self::Pow(base, n) => {
                if *n == 0 {
                    return Some(LinearExpr::one());
                }
                let base = base.extract_linear()?;
                if let Some(base) = base.as_constant() {
                    return Some(LinearExpr::constant(pow(base.clone(), *n)));
                }
                if *n == 1 {
                    Some(base)
                } else {
                    None
                }
            }
        }
    }

    pub fn to_rational_function(&self) -> RationalFunction<T>
    where
        T: Clone
            + PartialEq
            + Zero
            + One
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>,
    {
        match self {
            Self::Constant(c) => RationalFunction::constant(c.clone()),
            Self::Variable(i) => RationalFunction::var(*i),
            Self::Add(lhs, rhs) => lhs.to_rational_function() + rhs.to_rational_function(),
            Self::Mul(lhs, rhs) => lhs.to_rational_function() * rhs.to_rational_function(),
            Self::Pow(base, n) => base.to_rational_function().pow(*n),
        }
    }

    fn eval_dual(&self, values: &[T], var: usize) -> (T, T)
    where
        T: Clone
            + From<i32>
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>
            + One
            + Zero,
    {
        match self {
            Self::Constant(c) => (c.clone(), T::zero()),
            Self::Variable(v) => {
                if *v == var {
                    (values[*v].clone(), T::one())
                } else {
                    (values[*v].clone(), T::zero())
                }
            }
            Self::Add(lhs, rhs) => {
                let (lhs_val, lhs_grad) = lhs.eval_dual(values, var);
                let (rhs_val, rhs_grad) = rhs.eval_dual(values, var);
                (lhs_val + rhs_val, lhs_grad + rhs_grad)
            }
            Self::Mul(lhs, rhs) => {
                let (lhs_val, lhs_grad) = lhs.eval_dual(values, var);
                let (rhs_val, rhs_grad) = rhs.eval_dual(values, var);
                (
                    lhs_val.clone() * rhs_val.clone(),
                    lhs_grad * rhs_val + rhs_grad * lhs_val,
                )
            }
            Self::Pow(base, n) => {
                if n == &0 {
                    (T::one(), T::zero())
                } else {
                    let (base_val, base_grad) = base.eval_dual(values, var);
                    let outer_deriv = pow(base_val.clone(), *n - 1);
                    let grad = base_grad * outer_deriv.clone() * T::from(*n);
                    (outer_deriv * base_val, grad)
                }
            }
        }
    }

    pub fn derivative_at(&self, values: &[T], var: usize) -> T
    where
        T: Clone
            + From<i32>
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>
            + One
            + Zero,
    {
        self.eval_dual(values, var).1
    }

    pub fn gradient_at(&self, values: &[T]) -> Vec<T>
    where
        T: Clone
            + From<i32>
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>
            + One
            + Zero,
    {
        (0..values.len())
            .map(|i| self.derivative_at(values, i))
            .collect()
    }

    pub fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &T) -> z3::ast::Real<'a>,
    ) -> z3::ast::Real<'a> {
        match self {
            Self::Constant(f) => conv(ctx, f),
            Self::Variable(v) => z3::ast::Real::new_const(ctx, *v as u32),
            Self::Add(e1, e2) => e1.to_z3(ctx, conv) + e2.to_z3(ctx, conv),
            Self::Mul(e1, e2) => e1.to_z3(ctx, conv) * e2.to_z3(ctx, conv),
            Self::Pow(e, n) => e
                .to_z3(ctx, conv)
                .power(&z3::ast::Int::from_i64(ctx, (*n).into()).to_real()),
        }
    }

    pub fn to_python(&self) -> String
    where
        T: Display,
    {
        match self {
            Self::Constant(c) => c.to_string(),
            Self::Variable(v) => format!("x[{v}]"),
            Self::Add(lhs, rhs) => format!("({} + {})", lhs.to_python(), rhs.to_python()),
            Self::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python(), rhs.to_python()),
            Self::Pow(lhs, rhs) => format!("({} ** {})", lhs.to_python(), rhs),
        }
    }

    pub fn to_z3_string(&self) -> String
    where
        T: Display + Clone + Neg<Output = T> + Zero + One + PartialOrd,
    {
        match self {
            Self::Constant(value) => {
                if value < &T::zero() {
                    format!("(- {})", -value.clone())
                } else {
                    format!("{value}")
                }
            }
            Self::Variable(i) => format!("x{i}"),
            Self::Add(lhs, rhs) => format!("(+ {lhs} {rhs})"),
            Self::Mul(lhs, rhs) => {
                if Self::Constant(-T::zero()) == **rhs {
                    format!("(- {lhs})")
                } else {
                    format!("(* {lhs} {rhs})")
                }
            }
            Self::Pow(expr, n) => {
                if *n == -1 {
                    format!("(/ 1 {expr})")
                } else if *n < 0 {
                    format!("(/ 1 (^ {expr} {}))", -n)
                } else {
                    format!("(^ {expr} {n})")
                }
            }
        }
    }

    pub fn to_python_z3(&self) -> String
    where
        T: Display,
    {
        match self {
            Self::Constant(c) => c.to_string(),
            Self::Variable(v) => format!("x{v}"),
            Self::Add(lhs, rhs) => format!("({} + {})", lhs.to_python_z3(), rhs.to_python_z3()),
            Self::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python_z3(), rhs.to_python_z3()),
            Self::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_python_z3(), rhs),
        }
    }

    pub fn to_qepcad(&self, conv: &impl Fn(&T) -> String) -> String {
        match self {
            Self::Constant(c) => conv(c),
            Self::Variable(v) => format!("{}", crate::ppl::Var(*v)),
            Self::Add(lhs, rhs) => format!("({} + {})", lhs.to_qepcad(conv), rhs.to_qepcad(conv)),
            Self::Mul(lhs, rhs) => format!("({} {})", lhs.to_qepcad(conv), rhs.to_qepcad(conv)),
            Self::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_qepcad(conv), rhs),
        }
    }

    pub fn eval(&self, values: &[T]) -> T
    where
        T: Clone
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>
            + One
            + Zero,
    {
        match self {
            Self::Constant(c) => c.clone(),
            Self::Variable(v) => values[*v].clone(),
            Self::Add(lhs, rhs) => lhs.eval(values) + rhs.eval(values),
            Self::Mul(lhs, rhs) => lhs.eval(values) * rhs.eval(values),
            Self::Pow(base, n) => pow(base.eval(values), *n),
        }
    }
}

impl SymExpr<f64> {
    pub fn eval_exact(&self, values: &[f64]) -> Rational {
        match self {
            Self::Constant(c) => Rational::from_f64(*c).unwrap(),
            Self::Variable(v) => Rational::from_f64(values[*v]).unwrap(),
            Self::Add(lhs, rhs) => lhs.eval_exact(values) + rhs.eval_exact(values),
            Self::Mul(lhs, rhs) => lhs.eval_exact(values) * rhs.eval_exact(values),
            Self::Pow(base, n) => base.eval_exact(values).pow(*n),
        }
    }
}

impl<T: From<f64>> From<f64> for SymExpr<T> {
    fn from(value: f64) -> Self {
        Self::Constant(value.into())
    }
}

impl<T: Zero> Zero for SymExpr<T> {
    fn zero() -> Self {
        Self::Constant(T::zero())
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_zero(),
            _ => false,
        }
    }
}

impl<T: Zero + One> One for SymExpr<T>
where
    T: One + PartialEq,
{
    fn one() -> Self {
        Self::Constant(T::one())
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_one(),
            _ => false,
        }
    }
}

impl<T: Neg> Neg for SymExpr<T>
where
    T: Clone + Zero + One + Neg<Output = T>,
    Self: Mul<Output = Self>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else if let Self::Constant(c) = self {
            Self::Constant(-c.clone())
        } else {
            self * Self::Constant(-T::one())
        }
    }
}

impl<T> Add for SymExpr<T>
where
    T: Zero + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            Self::Add(Rc::new(self), Rc::new(rhs))
        }
    }
}

impl<T> AddAssign for SymExpr<T>
where
    Self: Clone + Add<Output = Self> + Zero,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<T> Sub for SymExpr<T>
where
    Self: PartialEq + Zero + Add<Output = Self> + Neg<Output = Self>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self == rhs {
            Self::zero()
        } else {
            self + (-rhs)
        }
    }
}

impl<T> SubAssign for SymExpr<T>
where
    Self: Clone + Sub<Output = Self>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<T> Mul for SymExpr<T>
where
    Self: Zero + One + PartialEq,
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
            Self::Mul(Rc::new(self), Rc::new(rhs))
        }
    }
}

impl<T> MulAssign for SymExpr<T>
where
    Self: Clone + Mul<Output = Self>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T> Div for SymExpr<T>
where
    T: PartialEq,
    Self: Mul<Output = Self> + PartialEq + Zero + One + Clone,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl<T> DivAssign for SymExpr<T>
where
    Self: Div<Output = Self> + Clone,
{
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<T> std::fmt::Display for SymExpr<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(c) => write!(f, "{c}"),
            Self::Variable(v) => write!(f, "x{v}"),
            Self::Add(lhs, rhs) => write!(f, "({lhs} + {rhs})"),
            Self::Mul(lhs, rhs) => write!(f, "({lhs} * {rhs})"),
            Self::Pow(base, n) => write!(f, "({base} ^ {n})"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SymConstraint<T> {
    Eq(SymExpr<T>, SymExpr<T>),
    Lt(SymExpr<T>, SymExpr<T>),
    Le(SymExpr<T>, SymExpr<T>),
    Or(Vec<SymConstraint<T>>),
}
impl<T> SymConstraint<T> {
    pub fn or(constraints: Vec<SymConstraint<T>>) -> Self {
        Self::Or(constraints)
    }

    pub fn is_trivial(&self) -> bool
    where
        T: PartialEq + Zero,
    {
        match self {
            SymConstraint::Eq(lhs, rhs) | SymConstraint::Le(lhs, rhs) => lhs == rhs,
            SymConstraint::Lt(..) => false,
            SymConstraint::Or(constraints) => constraints.iter().any(SymConstraint::is_trivial),
        }
    }

    pub fn to_poly(&self) -> PolyConstraint<T>
    where
        T: Clone
            + PartialEq
            + Zero
            + One
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>,
    {
        match self {
            SymConstraint::Eq(lhs, rhs) => {
                let lhs = lhs.to_rational_function();
                let rhs = rhs.to_rational_function();
                PolyConstraint::Eq(lhs.numer * rhs.denom, rhs.numer * lhs.denom)
            }
            SymConstraint::Lt(lhs, rhs) => {
                let lhs = lhs.to_rational_function();
                let rhs = rhs.to_rational_function();
                PolyConstraint::Lt(lhs.numer * rhs.denom, rhs.numer * lhs.denom)
            }
            SymConstraint::Le(lhs, rhs) => {
                let lhs = lhs.to_rational_function();
                let rhs = rhs.to_rational_function();
                PolyConstraint::Le(lhs.numer * rhs.denom, rhs.numer * lhs.denom)
            }
            SymConstraint::Or(constraints) => {
                PolyConstraint::or(constraints.iter().map(SymConstraint::to_poly).collect())
            }
        }
    }

    pub fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &T) -> z3::ast::Real<'a>,
    ) -> z3::ast::Bool<'a> {
        match self {
            SymConstraint::Eq(e1, e2) => {
                z3::ast::Ast::_eq(&e1.to_z3(ctx, conv), &e2.to_z3(ctx, conv))
            }
            SymConstraint::Lt(e1, e2) => e1.to_z3(ctx, conv).lt(&e2.to_z3(ctx, conv)),
            SymConstraint::Le(e1, e2) => e1.to_z3(ctx, conv).le(&e2.to_z3(ctx, conv)),
            SymConstraint::Or(constraints) => {
                let disjuncts = constraints
                    .iter()
                    .map(|c| c.to_z3(ctx, conv))
                    .collect::<Vec<_>>();
                z3::ast::Bool::or(ctx, &disjuncts.iter().collect::<Vec<_>>())
            }
        }
    }

    pub fn to_qepcad(&self, conv: &impl Fn(&T) -> String) -> String {
        match self {
            SymConstraint::Eq(lhs, rhs) => {
                format!("{} = {}", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymConstraint::Lt(lhs, rhs) => {
                format!("{} < {}", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymConstraint::Le(lhs, rhs) => {
                format!("{} <= {}", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymConstraint::Or(cs) => {
                let mut res = "[".to_owned();
                let mut first = true;
                for c in cs {
                    if first {
                        first = false;
                    } else {
                        res += r" \/ ";
                    }
                    res += &c.to_qepcad(conv);
                }
                res + "]"
            }
        }
    }

    pub fn to_python_z3(&self) -> String
    where
        T: Display,
    {
        match self {
            SymConstraint::Eq(lhs, rhs) => {
                format!("{} == {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymConstraint::Lt(lhs, rhs) => {
                format!("{} < {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymConstraint::Le(lhs, rhs) => {
                format!("{} <= {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymConstraint::Or(cs) => {
                let mut res = "Or(".to_owned();
                let mut first = true;
                for c in cs {
                    if first {
                        first = false;
                    } else {
                        res += ", ";
                    }
                    res += &c.to_python_z3();
                }
                res + ")"
            }
        }
    }

    pub fn substitute(&self, replacements: &[SymExpr<T>]) -> SymConstraint<T>
    where
        T: PartialEq,
        SymExpr<T>: Clone + Zero + One + Add<Output = SymExpr<T>> + Mul<Output = SymExpr<T>>,
    {
        match self {
            SymConstraint::Eq(e1, e2) => {
                SymConstraint::Eq(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Lt(e1, e2) => {
                SymConstraint::Lt(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Le(e1, e2) => {
                SymConstraint::Le(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Or(constraints) => SymConstraint::Or(
                constraints
                    .iter()
                    .map(|c| c.substitute(replacements))
                    .collect(),
            ),
        }
    }

    pub fn gradient_at(&self, values: &[T]) -> Vec<T>
    where
        T: Clone
            + From<i32>
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Mul<Output = T>
            + Div<Output = T>
            + One
            + Zero,
        SymExpr<T>: Sub<Output = SymExpr<T>>,
    {
        match self {
            SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                let term = rhs.clone() - lhs.clone();
                term.gradient_at(values)
            }
            _ => vec![T::zero(); values.len()],
        }
    }

    pub fn extract_linear(&self) -> Option<LinearConstraint<T>>
    where
        T: Clone
            + PartialOrd
            + Zero
            + One
            + Neg<Output = T>
            + AddAssign
            + Add<Output = T>
            + MulAssign
            + Div<Output = T>,
    {
        match self {
            SymConstraint::Eq(e1, e2) => Some(LinearConstraint::eq(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Lt(e1, e2) | SymConstraint::Le(e1, e2) => Some(LinearConstraint::le(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Or(constraints) => {
                // Here we only support constraints without variables
                for constraint in constraints {
                    if let Some(linear_constraint) = constraint.extract_linear() {
                        if linear_constraint.eval_constant() == Some(true) {
                            return Some(LinearConstraint::eq(
                                LinearExpr::constant(T::zero()),
                                LinearExpr::constant(T::zero()),
                            ));
                        }
                    }
                }
                None
            }
        }
    }
}

impl SymConstraint<f64> {
    pub fn holds_exact(&self, values: &[f64]) -> bool {
        match self {
            SymConstraint::Lt(lhs, rhs) => lhs.eval_exact(values) < rhs.eval_exact(values),
            SymConstraint::Le(lhs, rhs) => lhs.eval_exact(values) <= rhs.eval_exact(values),
            _ => true,
        }
    }
    pub fn is_close(&self, point: &[f64], min_dist: f64) -> bool {
        match self {
            SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                let term = lhs.clone() - rhs.clone();
                let val = term.eval(point);
                let grad = term.gradient_at(point);
                let grad_len_sq = grad
                    .iter()
                    .map(|g| g.clone() * g.clone())
                    .fold(0.0, |acc, f| acc + f);
                if grad_len_sq.is_zero() {
                    return false;
                }
                let dist = -val.clone() / grad_len_sq.sqrt();
                dist < min_dist
            }
            _ => false,
        }
    }
    pub fn estimate_signed_dist(&self, point: &[f64]) -> f64 {
        match self {
            SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                let term = lhs.clone() - rhs.clone();
                let val = term.eval(point);
                let grad = term.gradient_at(point);
                if val == 0.0 {
                    return 0.0;
                }
                val / norm(&ArrayView1::from(&grad))
            }
            _ => -1.0,
        }
    }
}

impl<T> std::fmt::Display for SymConstraint<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(e1, e2) => write!(f, "{e1} = {e2}"),
            Self::Lt(e1, e2) => write!(f, "{e1} < {e2}"),
            Self::Le(e1, e2) => write!(f, "{e1} <= {e2}"),
            Self::Or(constraints) => {
                write!(f, "(")?;
                for (i, c) in constraints.iter().enumerate() {
                    if i > 0 {
                        write!(f, " OR ")?;
                    }
                    write!(f, "{c}")?;
                }
                write!(f, ")")
            }
        }
    }
}
