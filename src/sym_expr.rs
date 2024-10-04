use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use ndarray::ArrayView1;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

use crate::{
    numbers::{FloatRat, Rational},
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
    pub fn into_expr(self) -> SymExpr {
        SymExpr(Rc::new(self))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymExpr(Rc<SymExprKind>);

impl SymExpr {
    pub(crate) fn kind(&self) -> &SymExprKind {
        self.0.as_ref()
    }

    /// To estimate the memory usage of the expression
    pub fn count_nodes(&self) -> usize {
        self.count_nodes_with(&mut FxHashMap::default())
    }

    fn count_nodes_with(&self, cache: &mut FxHashMap<usize, usize>) -> usize {
        let key = self.0.as_ref() as *const SymExprKind as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(_) = cache.get(&key) {
                return 1;
            }
            cache.insert(key, 1);
        }
        match self.kind() {
            SymExprKind::Constant(_) => 1,
            SymExprKind::Variable(_) => 1,
            SymExprKind::Add(lhs, rhs) | SymExprKind::Mul(lhs, rhs) => {
                1 + lhs.count_nodes_with(cache) + rhs.count_nodes_with(cache)
            }
            SymExprKind::Pow(base, _) => 1 + base.count_nodes_with(cache),
        }
    }

    pub fn var(i: usize) -> Self {
        SymExprKind::Variable(i).into_expr()
    }

    pub fn inverse(self) -> Self {
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

    /// Must equal `rhs`.
    pub fn must_eq(self, rhs: Self) -> SymConstraint {
        SymConstraint::Eq(self, rhs)
    }

    /// Must be less than `rhs`.
    pub fn must_lt(self, rhs: Self) -> SymConstraint {
        SymConstraint::Lt(self, rhs)
    }

    /// Must be less than or equal to `rhs`.
    pub fn must_le(self, rhs: Self) -> SymConstraint {
        SymConstraint::Le(self, rhs)
    }

    /// Must be greater than `rhs`.
    pub fn must_gt(self, rhs: Self) -> SymConstraint {
        SymConstraint::Lt(rhs, self)
    }

    /// Must be greater than or equal to `rhs`.
    pub fn must_ge(self, rhs: Self) -> SymConstraint {
        SymConstraint::Le(rhs, self)
    }

    pub fn substitute_with(
        &self,
        replacements: &[Self],
        cache: &mut FxHashMap<usize, Self>,
    ) -> Self {
        let key = self.0.as_ref() as *const SymExprKind as usize;
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

    pub fn extract_linear(&self) -> Option<LinearExpr> {
        match self.kind() {
            SymExprKind::Constant(c) => Some(LinearExpr::constant(c.clone())),
            SymExprKind::Variable(i) => Some(LinearExpr::var(*i)),
            SymExprKind::Add(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                Some(lhs + rhs)
            }
            SymExprKind::Mul(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
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

    fn eval_dual(&self, values: &[f64], var: usize) -> (f64, f64) {
        match self.kind() {
            SymExprKind::Constant(c) => (c.float(), 0.0),
            SymExprKind::Variable(v) => {
                if *v == var {
                    (values[*v], 1.0)
                } else {
                    (values[*v], 0.0)
                }
            }
            SymExprKind::Add(lhs, rhs) => {
                let (lhs_val, lhs_grad) = lhs.eval_dual(values, var);
                let (rhs_val, rhs_grad) = rhs.eval_dual(values, var);
                (lhs_val + rhs_val, lhs_grad + rhs_grad)
            }
            SymExprKind::Mul(lhs, rhs) => {
                let (lhs_val, lhs_grad) = lhs.eval_dual(values, var);
                let (rhs_val, rhs_grad) = rhs.eval_dual(values, var);
                (lhs_val * rhs_val, lhs_grad * rhs_val + rhs_grad * lhs_val)
            }
            SymExprKind::Pow(base, n) => {
                if n == &0 {
                    (1.0, 0.0)
                } else {
                    let (base_val, base_grad) = base.eval_dual(values, var);
                    let outer_deriv = pow(base_val, *n - 1);
                    let grad = base_grad * outer_deriv * f64::from(*n);
                    (outer_deriv * base_val, grad)
                }
            }
        }
    }

    pub fn derivative_at(&self, values: &[f64], var: usize) -> f64 {
        self.eval_dual(values, var).1
    }

    pub fn gradient_at(&self, values: &[f64]) -> Vec<f64> {
        (0..values.len())
            .map(|i| self.derivative_at(values, i))
            .collect()
    }

    pub fn to_z3<'a>(
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

    pub fn to_python(&self) -> String {
        match self.kind() {
            SymExprKind::Constant(c) => c.to_string(),
            SymExprKind::Variable(v) => format!("x[{v}]"),
            SymExprKind::Add(lhs, rhs) => format!("({} + {})", lhs.to_python(), rhs.to_python()),
            SymExprKind::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python(), rhs.to_python()),
            SymExprKind::Pow(lhs, rhs) => format!("({} ** {})", lhs.to_python(), rhs),
        }
    }

    pub fn to_z3_string(&self) -> String {
        match self.kind() {
            SymExprKind::Constant(value) => {
                if value.rat() < Rational::zero() {
                    format!("(- {})", -value.clone())
                } else {
                    format!("{value}")
                }
            }
            SymExprKind::Variable(i) => format!("x{i}"),
            SymExprKind::Add(lhs, rhs) => format!("(+ {lhs} {rhs})"),
            SymExprKind::Mul(lhs, rhs) => {
                if SymExpr::from(-Rational::one()) == *rhs {
                    format!("(- {lhs})")
                } else {
                    format!("(* {lhs} {rhs})")
                }
            }
            SymExprKind::Pow(expr, n) => {
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

    pub fn to_python_z3(&self) -> String {
        match self.kind() {
            SymExprKind::Constant(c) => c.to_string(),
            SymExprKind::Variable(v) => format!("x{v}"),
            SymExprKind::Add(lhs, rhs) => {
                format!("({} + {})", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymExprKind::Mul(lhs, rhs) => {
                format!("({} * {})", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymExprKind::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_python_z3(), rhs),
        }
    }

    pub fn to_qepcad(&self, conv: &impl Fn(&Rational) -> String) -> String {
        match self.kind() {
            SymExprKind::Constant(c) => conv(&c.rat()),
            SymExprKind::Variable(v) => format!("{}", crate::ppl::Var(*v)),
            SymExprKind::Add(lhs, rhs) => {
                format!("({} + {})", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymExprKind::Mul(lhs, rhs) => {
                format!("({} {})", lhs.to_qepcad(conv), rhs.to_qepcad(conv))
            }
            SymExprKind::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_qepcad(conv), rhs),
        }
    }

    pub fn eval_float(&self, values: &[f64]) -> f64 {
        match self.kind() {
            SymExprKind::Constant(c) => c.float(),
            SymExprKind::Variable(v) => values[*v],
            SymExprKind::Add(lhs, rhs) => lhs.eval_float(values) + rhs.eval_float(values),
            SymExprKind::Mul(lhs, rhs) => lhs.eval_float(values) * rhs.eval_float(values),
            SymExprKind::Pow(base, n) => pow(base.eval_float(values), *n),
        }
    }

    pub fn eval_exact(&self, values: &[Rational]) -> Rational {
        match self.kind() {
            SymExprKind::Constant(c) => c.rat().clone(),
            SymExprKind::Variable(v) => values[*v].clone(),
            SymExprKind::Add(lhs, rhs) => lhs.eval_exact(values) + rhs.eval_exact(values),
            SymExprKind::Mul(lhs, rhs) => lhs.eval_exact(values) * rhs.eval_exact(values),
            SymExprKind::Pow(base, n) => base.eval_exact(values).pow(*n),
        }
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
pub enum SymConstraint {
    Eq(SymExpr, SymExpr),
    Lt(SymExpr, SymExpr),
    Le(SymExpr, SymExpr),
    Or(Vec<SymConstraint>),
}
impl SymConstraint {
    pub fn or(constraints: Vec<SymConstraint>) -> Self {
        Self::Or(constraints)
    }

    pub fn is_trivial(&self) -> bool {
        match self {
            SymConstraint::Eq(lhs, rhs) | SymConstraint::Le(lhs, rhs) => lhs == rhs,
            SymConstraint::Lt(..) => false,
            SymConstraint::Or(constraints) => constraints.iter().any(SymConstraint::is_trivial),
        }
    }

    pub fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &Rational) -> z3::ast::Real<'a>,
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

    pub fn to_qepcad(&self, conv: &impl Fn(&Rational) -> String) -> String {
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

    pub fn to_python_z3(&self) -> String {
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

    pub fn substitute_with(
        &self,
        replacements: &[SymExpr],
        cache: &mut FxHashMap<usize, SymExpr>,
    ) -> SymConstraint {
        match self {
            SymConstraint::Eq(e1, e2) => SymConstraint::Eq(
                e1.substitute_with(replacements, cache),
                e2.substitute_with(replacements, cache),
            ),
            SymConstraint::Lt(e1, e2) => SymConstraint::Lt(
                e1.substitute_with(replacements, cache),
                e2.substitute_with(replacements, cache),
            ),
            SymConstraint::Le(e1, e2) => SymConstraint::Le(
                e1.substitute_with(replacements, cache),
                e2.substitute_with(replacements, cache),
            ),
            SymConstraint::Or(constraints) => SymConstraint::Or(
                constraints
                    .iter()
                    .map(|c| c.substitute_with(replacements, cache))
                    .collect(),
            ),
        }
    }

    pub fn gradient_at(&self, values: &[f64]) -> Vec<f64> {
        match self {
            SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                let term = rhs.clone() - lhs.clone();
                term.gradient_at(values)
            }
            _ => vec![0.0; values.len()],
        }
    }

    pub fn extract_linear(&self) -> Option<LinearConstraint> {
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
                                LinearExpr::constant(FloatRat::zero()),
                                LinearExpr::constant(FloatRat::zero()),
                            ));
                        }
                    }
                }
                None
            }
        }
    }

    pub fn holds_exact_f64(&self, values: &[f64]) -> bool {
        let values = values
            .iter()
            .map(|r| Rational::from(*r))
            .collect::<Vec<_>>();
        self.holds_exact(&values)
    }

    pub fn holds_exact(&self, values: &[Rational]) -> bool {
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
                let val = term.eval_float(point);
                let grad = term.gradient_at(point);
                let grad_len_sq = grad.iter().map(|g| g * g).fold(0.0, |acc, f| acc + f);
                if grad_len_sq.is_zero() {
                    return false;
                }
                let dist = -val / grad_len_sq.sqrt();
                dist < min_dist
            }
            _ => false,
        }
    }
    pub fn estimate_signed_dist(&self, point: &[f64]) -> f64 {
        match self {
            SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                let term = lhs.clone() - rhs.clone();
                let val = term.eval_float(point);
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

impl std::fmt::Display for SymConstraint {
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
