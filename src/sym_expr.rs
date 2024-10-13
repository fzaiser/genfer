use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use descent::expr::{dynam::Expr, Var};
use ndarray::ArrayView1;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

use crate::{
    numbers::{FloatNumber, FloatRat, Rational},
    solvers::linear::{LinearConstraint, LinearExpr},
    util::{norm, pow},
};

#[derive(Debug, Clone, PartialEq)]
pub enum SymExprKind {
    Constant(FloatRat),
    Variable(usize),
    Add(SymExpr, SymExpr),
    Mul(SymExpr, SymExpr),
    Pow(SymExpr, i32),
}

impl SymExprKind {
    pub(crate) fn into_expr(self) -> SymExpr {
        SymExpr(Rc::new(self))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymExpr(Rc<SymExprKind>);

impl SymExpr {
    pub(crate) fn kind(&self) -> &SymExprKind {
        self.0.as_ref()
    }

    pub(crate) fn var(i: usize) -> Self {
        SymExprKind::Variable(i).into_expr()
    }

    pub(crate) fn inverse(self) -> Self {
        self.pow(-1)
    }

    pub fn pow(self, exp: i32) -> Self {
        if exp == 0 {
            Self::one()
        } else if exp == 1 || (exp >= 0 && self.is_zero()) || self.is_one() {
            self
        } else if let SymExprKind::Constant(c) = self.kind() {
            SymExprKind::Constant(c.pow(exp)).into_expr()
        } else {
            SymExprKind::Pow(self, exp).into_expr()
        }
    }

    /// Must be less than or equal to `rhs`.
    pub(crate) fn must_le(self, rhs: Self) -> SymConstraint {
        SymConstraint { lhs: self, rhs }
    }

    pub(crate) fn substitute_with(
        &self,
        replacements: &[Self],
        cache: &mut FxHashMap<usize, Self>,
    ) -> Self {
        let key = std::ptr::from_ref::<SymExprKind>(self.0.as_ref()) as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return cached_result.clone();
            }
        }
        let result = match self.kind() {
            SymExprKind::Constant(_) => self.clone(),
            SymExprKind::Variable(i) => replacements[*i].clone(),
            SymExprKind::Add(lhs, rhs) => {
                lhs.substitute_with(replacements, cache) + rhs.substitute_with(replacements, cache)
            }
            SymExprKind::Mul(lhs, rhs) => {
                lhs.substitute_with(replacements, cache) * rhs.substitute_with(replacements, cache)
            }
            SymExprKind::Pow(base, n) => base.substitute_with(replacements, cache).pow(*n),
        };
        // Only cache the result if it is shared:
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }

    pub fn extract_constant(&self) -> Option<FloatRat> {
        match self.kind() {
            SymExprKind::Constant(c) => Some(c.clone()),
            _ => None,
        }
    }

    pub(crate) fn extract_linear(
        &self,
        cache: &mut FxHashMap<usize, Option<LinearExpr>>,
    ) -> Option<LinearExpr> {
        let key = std::ptr::from_ref::<SymExprKind>(self.0.as_ref()) as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return cached_result.clone();
            }
        }
        let result = match self.kind() {
            SymExprKind::Constant(c) => Some(LinearExpr::constant(c.rat().clone())),
            SymExprKind::Variable(i) => Some(LinearExpr::var(*i)),
            SymExprKind::Add(lhs, rhs) => {
                let lhs = lhs.extract_linear(cache)?;
                let rhs = rhs.extract_linear(cache)?;
                Some(lhs + rhs)
            }
            SymExprKind::Mul(lhs, rhs) => {
                let lhs = lhs.extract_linear(cache)?;
                let rhs = rhs.extract_linear(cache)?;
                if let Some(factor) = lhs.as_constant() {
                    Some(rhs * factor.clone())
                } else {
                    rhs.as_constant().map(|factor| lhs * factor.clone())
                }
            }
            SymExprKind::Pow(base, n) => {
                if *n == 0 {
                    return Some(LinearExpr::one());
                }
                let base = base.extract_linear(cache)?;
                if let Some(base) = base.as_constant() {
                    return Some(LinearExpr::constant(pow(base.clone(), *n)));
                }
                if *n == 1 {
                    Some(base)
                } else {
                    None
                }
            }
        };
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }

    fn eval_dual(
        &self,
        values: &[f64],
        var: usize,
        cache: &mut FxHashMap<usize, (f64, f64)>,
    ) -> (f64, f64) {
        let key = std::ptr::from_ref::<SymExprKind>(self.0.as_ref()) as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return *cached_result;
            }
        }
        let result = match self.kind() {
            SymExprKind::Constant(c) => (c.float(), 0.0),
            SymExprKind::Variable(v) => {
                if *v == var {
                    (values[*v], 1.0)
                } else {
                    (values[*v], 0.0)
                }
            }
            SymExprKind::Add(lhs, rhs) => {
                let (lhs_val, lhs_grad) = lhs.eval_dual(values, var, cache);
                let (rhs_val, rhs_grad) = rhs.eval_dual(values, var, cache);
                (lhs_val + rhs_val, lhs_grad + rhs_grad)
            }
            SymExprKind::Mul(lhs, rhs) => {
                let (lhs_val, lhs_grad) = lhs.eval_dual(values, var, cache);
                let (rhs_val, rhs_grad) = rhs.eval_dual(values, var, cache);
                (lhs_val * rhs_val, lhs_grad * rhs_val + rhs_grad * lhs_val)
            }
            SymExprKind::Pow(base, n) => {
                if n == &0 {
                    (1.0, 0.0)
                } else {
                    let (base_val, base_grad) = base.eval_dual(values, var, cache);
                    let outer_deriv = pow(base_val, *n - 1);
                    let grad = base_grad * outer_deriv * f64::from(*n);
                    (outer_deriv * base_val, grad)
                }
            }
        };
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result);
        }
        result
    }

    pub(crate) fn derivative_at(
        &self,
        values: &[f64],
        var: usize,
        cache: &mut FxHashMap<usize, (f64, f64)>,
    ) -> f64 {
        self.eval_dual(values, var, cache).1
    }

    pub(crate) fn gradient_at(
        &self,
        values: &[f64],
        grad_cache: &mut [FxHashMap<usize, (f64, f64)>],
    ) -> Vec<f64> {
        // TODO: reverse-mode AD would be faster
        (0..values.len())
            .map(|i| self.derivative_at(values, i, &mut grad_cache[i]))
            .collect()
    }

    pub(crate) fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &Rational) -> z3::ast::Real<'a>,
    ) -> z3::ast::Real<'a> {
        match self.kind() {
            SymExprKind::Constant(f) => conv(ctx, &f.rat()),
            SymExprKind::Variable(v) => z3::ast::Real::new_const(ctx, *v as u32),
            SymExprKind::Add(e1, e2) => e1.to_z3(ctx, conv) + e2.to_z3(ctx, conv),
            SymExprKind::Mul(e1, e2) => e1.to_z3(ctx, conv) * e2.to_z3(ctx, conv),
            SymExprKind::Pow(e, n) => e
                .to_z3(ctx, conv)
                .power(&z3::ast::Int::from_i64(ctx, (*n).into()).to_real()),
        }
    }

    pub(crate) fn to_smtlib(&self) -> String {
        match self.kind() {
            SymExprKind::Constant(value) => {
                let value = value.rat();
                if value.is_finite() {
                    let (numer, denom) = value.to_integer_ratio();
                    if denom == 1 {
                        if numer.is_negative() {
                            format!("(- {})", -numer)
                        } else {
                            format!("{numer}")
                        }
                    } else if numer.is_negative() {
                        format!("(- (/ {} {denom}))", -numer)
                    } else {
                        format!("(/ {numer} {denom})")
                    }
                } else {
                    panic!("Nonfinite number {value} cannot be represented in SMT-LIB");
                }
            }
            SymExprKind::Variable(i) => format!("x{i}"),
            SymExprKind::Add(lhs, rhs) => format!("(+ {} {})", lhs.to_smtlib(), rhs.to_smtlib()),
            SymExprKind::Mul(lhs, rhs) => {
                if SymExpr::from(-Rational::one()) == *rhs {
                    format!("(- {})", lhs.to_smtlib())
                } else {
                    format!("(* {} {})", lhs.to_smtlib(), rhs.to_smtlib())
                }
            }
            SymExprKind::Pow(expr, n) => {
                if *n == -1 {
                    format!("(/ 1 {})", expr.to_smtlib())
                } else if *n < 0 {
                    format!(
                        "(/ 1 (* {}))",
                        vec![expr.to_smtlib(); (-n) as usize].join(" ")
                    )
                } else {
                    format!("(* {})", vec![expr.to_smtlib(); *n as usize].join(" "))
                }
            }
        }
    }

    pub(crate) fn to_qepcad(&self, conv: &impl Fn(&Rational) -> String) -> String {
        match self.kind() {
            SymExprKind::Constant(c) => conv(&c.rat()),
            SymExprKind::Variable(v) => format!("{}", SymExpr::var(*v)),
            SymExprKind::Add(lhs, rhs) => {
                format!("({} + {})", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymExprKind::Mul(lhs, rhs) => {
                format!("({} {})", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymExprKind::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_qepcad(conv), rhs),
        }
    }

    pub(crate) fn eval_float(&self, values: &[f64], cache: &mut FxHashMap<usize, f64>) -> f64 {
        let key = std::ptr::from_ref::<SymExprKind>(self.0.as_ref()) as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return *cached_result;
            }
        }
        let result = match self.kind() {
            SymExprKind::Constant(c) => c.float(),
            SymExprKind::Variable(v) => values[*v],
            SymExprKind::Add(lhs, rhs) => {
                lhs.eval_float(values, cache) + rhs.eval_float(values, cache)
            }
            SymExprKind::Mul(lhs, rhs) => {
                lhs.eval_float(values, cache) * rhs.eval_float(values, cache)
            }
            SymExprKind::Pow(base, n) => pow(base.eval_float(values, cache), *n),
        };
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result);
        }
        result
    }

    pub fn eval_exact(
        &self,
        values: &[Rational],
        cache: &mut FxHashMap<usize, Rational>,
    ) -> Rational {
        let key = std::ptr::from_ref::<SymExprKind>(self.0.as_ref()) as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return cached_result.clone();
            }
        }
        let result = match self.kind() {
            SymExprKind::Constant(c) => c.rat().clone(),
            SymExprKind::Variable(v) => values[*v].clone(),
            SymExprKind::Add(lhs, rhs) => {
                lhs.eval_exact(values, cache) + rhs.eval_exact(values, cache)
            }
            SymExprKind::Mul(lhs, rhs) => {
                lhs.eval_exact(values, cache) * rhs.eval_exact(values, cache)
            }
            SymExprKind::Pow(base, n) => base.eval_exact(values, cache).pow(*n),
        };
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }

    pub(crate) fn to_ipopt_expr(&self, vars: &[Var], cache: &mut FxHashMap<usize, Expr>) -> Expr {
        let key = std::ptr::from_ref::<SymExprKind>(self.0.as_ref()) as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return cached_result.clone();
            }
        }
        let result = match self.kind() {
            SymExprKind::Constant(c) => c.float().into(),
            SymExprKind::Variable(v) => vars[*v].into(),
            SymExprKind::Add(lhs, rhs) => {
                lhs.to_ipopt_expr(vars, cache) + rhs.to_ipopt_expr(vars, cache)
            }
            SymExprKind::Mul(lhs, rhs) => {
                lhs.to_ipopt_expr(vars, cache) * rhs.to_ipopt_expr(vars, cache)
            }
            SymExprKind::Pow(base, exp) => {
                use descent::expr::dynam::NumOps;
                base.to_ipopt_expr(vars, cache).powi(*exp)
            }
        };
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }
}

impl From<u64> for SymExpr {
    fn from(value: u64) -> Self {
        SymExprKind::Constant(FloatRat::from(value)).into_expr()
    }
}

impl From<Rational> for SymExpr {
    fn from(value: Rational) -> Self {
        SymExprKind::Constant(value.into()).into_expr()
    }
}

impl From<FloatRat> for SymExpr {
    fn from(value: FloatRat) -> Self {
        SymExprKind::Constant(value).into_expr()
    }
}

impl Zero for SymExpr {
    fn zero() -> Self {
        SymExprKind::Constant(FloatRat::zero()).into_expr()
    }

    fn is_zero(&self) -> bool {
        match self.kind() {
            SymExprKind::Constant(x) => x.is_zero(),
            _ => false,
        }
    }
}

impl One for SymExpr {
    fn one() -> Self {
        SymExprKind::Constant(FloatRat::one()).into_expr()
    }

    fn is_one(&self) -> bool {
        match self.kind() {
            SymExprKind::Constant(x) => x.is_one(),
            _ => false,
        }
    }
}

impl Neg for SymExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else if let SymExprKind::Constant(c) = self.kind() {
            SymExprKind::Constant(-c.clone()).into_expr()
        } else {
            SymExprKind::Constant(-FloatRat::one()).into_expr() * self
        }
    }
}

impl Add for SymExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            match (self.kind(), rhs.kind()) {
                (SymExprKind::Constant(c1), SymExprKind::Constant(c2)) => {
                    Self::from(c1.clone() + c2.clone())
                }
                _ => SymExprKind::Add(self, rhs).into_expr(),
            }
        }
    }
}

impl AddAssign for SymExpr
where
    Self: Clone + Add<Output = Self> + Zero,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl Sub for SymExpr
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

impl SubAssign for SymExpr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl Mul for SymExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else if self.is_one() {
            rhs
        } else if rhs.is_one() {
            self
        } else {
            match (self.kind(), rhs.kind()) {
                (SymExprKind::Constant(c1), SymExprKind::Constant(c2)) => {
                    Self::from(c1.clone() * c2.clone())
                }
                _ => SymExprKind::Mul(self, rhs).into_expr(),
            }
        }
    }
}

impl MulAssign for SymExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl Div for SymExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl DivAssign for SymExpr
where
    Self: Div<Output = Self> + Clone,
{
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl std::fmt::Display for SymExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind() {
            SymExprKind::Constant(c) => write!(f, "{c}"),
            SymExprKind::Variable(v) => write!(f, "x{v}"),
            SymExprKind::Add(lhs, rhs) => write!(f, "({lhs} + {rhs})"),
            SymExprKind::Mul(lhs, rhs) => write!(f, "({lhs} * {rhs})"),
            SymExprKind::Pow(base, n) => write!(f, "({base} ^ {n})"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SymConstraint {
    pub lhs: SymExpr,
    pub rhs: SymExpr,
}
impl SymConstraint {
    pub(crate) fn is_trivial(&self) -> bool {
        self.lhs.is_zero() || self.lhs == self.rhs
    }

    pub(crate) fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &Rational) -> z3::ast::Real<'a>,
    ) -> z3::ast::Bool<'a> {
        self.lhs.to_z3(ctx, conv).le(&self.rhs.to_z3(ctx, conv))
    }

    pub(crate) fn to_qepcad(&self, conv: &impl Fn(&Rational) -> String) -> String {
        format!(
            "{} <= {}",
            self.lhs.to_qepcad(conv),
            self.rhs.to_qepcad(conv)
        )
    }

    pub(crate) fn to_smtlib(&self) -> String {
        format!("(<= {} {})", self.lhs.to_smtlib(), self.rhs.to_smtlib())
    }

    pub(crate) fn substitute_with(
        &self,
        replacements: &[SymExpr],
        cache: &mut FxHashMap<usize, SymExpr>,
    ) -> SymConstraint {
        self.lhs
            .substitute_with(replacements, cache)
            .must_le(self.rhs.substitute_with(replacements, cache))
    }

    pub(crate) fn gradient_at(
        &self,
        values: &[f64],
        cache: &mut [FxHashMap<usize, (f64, f64)>],
    ) -> Vec<f64> {
        let term = self.rhs.clone() - self.lhs.clone();
        term.gradient_at(values, cache)
    }

    pub(crate) fn extract_linear(
        &self,
        cache: &mut FxHashMap<usize, Option<LinearExpr>>,
    ) -> Option<LinearConstraint> {
        Some(LinearConstraint::le(
            self.lhs.extract_linear(cache)?,
            self.rhs.extract_linear(cache)?,
        ))
    }

    pub(crate) fn holds_exact(
        &self,
        values: &[Rational],
        cache: &mut FxHashMap<usize, Rational>,
    ) -> bool {
        self.lhs.eval_exact(values, cache) <= self.rhs.eval_exact(values, cache)
    }

    pub(crate) fn estimate_signed_dist(
        &self,
        point: &[f64],
        cache: &mut FxHashMap<usize, f64>,
        grad_cache: &mut [FxHashMap<usize, (f64, f64)>],
    ) -> f64 {
        let term = self.lhs.clone() - self.rhs.clone();
        let val = term.eval_float(point, cache);
        let grad = term.gradient_at(point, grad_cache);
        if val == 0.0 {
            return 0.0;
        }
        val / norm(&ArrayView1::from(&grad))
    }
}

impl std::fmt::Display for SymConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} <= {}", self.lhs, self.rhs)
    }
}
