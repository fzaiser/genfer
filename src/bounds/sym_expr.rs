use num_traits::{One, Zero};

use crate::bounds::linear::*;

#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr {
    Constant(f64),
    Variable(usize),
    Add(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    Pow(Box<SymExpr>, i32),
}

impl SymExpr {
    pub fn var(i: usize) -> Self {
        Self::Variable(i)
    }

    pub fn inverse(self) -> Self {
        self.pow(-1)
    }

    pub fn pow(self, n: i32) -> Self {
        if n == 0 {
            Self::one()
        } else if n == 1 || (n >= 0 && self.is_zero()) || self.is_one() {
            self
        } else {
            Self::Pow(Box::new(self), n)
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

    pub fn substitute(&self, replacements: &[SymExpr]) -> Self {
        match self {
            SymExpr::Constant(_) => self.clone(),
            SymExpr::Variable(i) => replacements[*i].clone(),
            SymExpr::Add(lhs, rhs) => lhs.substitute(replacements) + rhs.substitute(replacements),
            SymExpr::Mul(lhs, rhs) => lhs.substitute(replacements) * rhs.substitute(replacements),
            SymExpr::Pow(base, n) => base.substitute(replacements).pow(*n),
        }
    }

    pub fn extract_constant(&self) -> Option<f64> {
        match self {
            SymExpr::Constant(c) => Some(*c),
            _ => None,
        }
    }

    pub fn extract_linear(&self) -> Option<LinearExpr> {
        match self {
            SymExpr::Constant(c) => Some(LinearExpr::constant(*c)),
            SymExpr::Variable(i) => Some(LinearExpr::var(*i)),
            SymExpr::Add(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                Some(lhs + rhs)
            }
            SymExpr::Mul(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                if let Some(factor) = lhs.as_constant() {
                    Some(rhs * factor)
                } else if let Some(factor) = rhs.as_constant() {
                    Some(lhs * factor)
                } else {
                    None
                }
            }
            SymExpr::Pow(base, n) => {
                if *n == 0 {
                    return Some(LinearExpr::constant(1.0));
                }
                let base = base.extract_linear()?;
                if let Some(base) = base.as_constant() {
                    return Some(LinearExpr::constant(base.powi(*n)));
                }
                if *n == 1 {
                    Some(base)
                } else {
                    None
                }
            }
        }
    }

    pub fn to_z3<'a>(&self, ctx: &'a z3::Context) -> z3::ast::Real<'a> {
        match self {
            SymExpr::Constant(f) => {
                if !f.is_finite() {
                    unreachable!("Non-finite f64 in constraint: {f}");
                }
                let bits: u64 = f.to_bits();
                let sign: i64 = if bits >> 63 == 0 { 1 } else { -1 };
                let mut exponent = ((bits >> 52) & 0x7ff) as i64;
                let mantissa = if exponent == 0 {
                    (bits & 0x000f_ffff_ffff_ffff) << 1
                } else {
                    (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
                } as i64;
                // Exponent bias + mantissa shift
                exponent -= 1023 + 52;
                let m = z3::ast::Int::from_i64(ctx, sign * mantissa).to_real();
                let two = z3::ast::Int::from_i64(ctx, 2).to_real();
                let e = z3::ast::Int::from_i64(ctx, exponent).to_real();
                m * two.power(&e)
            }
            SymExpr::Variable(v) => z3::ast::Real::new_const(ctx, *v as u32),
            SymExpr::Add(e1, e2) => e1.to_z3(ctx) + e2.to_z3(ctx),
            SymExpr::Mul(e1, e2) => e1.to_z3(ctx) * e2.to_z3(ctx),
            SymExpr::Pow(e, n) => e
                .to_z3(ctx)
                .power(&z3::ast::Int::from_i64(ctx, (*n).into()).to_real()),
        }
    }

    pub fn to_python(&self) -> String {
        match self {
            SymExpr::Constant(c) => c.to_string(),
            SymExpr::Variable(v) => format!("x[{v}]"),
            SymExpr::Add(lhs, rhs) => format!("({} + {})", lhs.to_python(), rhs.to_python()),
            SymExpr::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python(), rhs.to_python()),
            SymExpr::Pow(lhs, rhs) => format!("({} ** {})", lhs.to_python(), rhs),
        }
    }

    pub fn to_python_z3(&self) -> String {
        match self {
            SymExpr::Constant(c) => c.to_string(),
            SymExpr::Variable(v) => format!("x{v}"),
            SymExpr::Add(lhs, rhs) => format!("({} + {})", lhs.to_python_z3(), rhs.to_python_z3()),
            SymExpr::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python_z3(), rhs.to_python_z3()),
            SymExpr::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_python_z3(), rhs),
        }
    }

    pub fn eval(&self, values: &[f64]) -> f64 {
        match self {
            SymExpr::Constant(c) => *c,
            SymExpr::Variable(v) => values[*v],
            SymExpr::Add(lhs, rhs) => lhs.eval(values) + rhs.eval(values),
            SymExpr::Mul(lhs, rhs) => lhs.eval(values) * rhs.eval(values),
            SymExpr::Pow(base, n) => base.eval(values).powi(*n),
        }
    }
}

impl From<f64> for SymExpr {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

impl Zero for SymExpr {
    fn zero() -> Self {
        Self::Constant(0.0)
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_zero(),
            _ => false,
        }
    }
}

impl One for SymExpr {
    fn one() -> Self {
        Self::Constant(1.0)
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_one(),
            _ => false,
        }
    }
}

impl std::ops::Neg for SymExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else if let Self::Constant(c) = self {
            (-c).into()
        } else {
            self * (-1.0).into()
        }
    }
}

impl std::ops::Add for SymExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            Self::Add(Box::new(self), Box::new(rhs))
        }
    }
}

impl std::ops::AddAssign for SymExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Sub for SymExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self == rhs {
            Self::zero()
        } else {
            self + (-rhs)
        }
    }
}

impl std::ops::SubAssign for SymExpr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl std::ops::Mul for SymExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else if self.is_one() {
            rhs
        } else if rhs.is_one() {
            self
        } else {
            Self::Mul(Box::new(self), Box::new(rhs))
        }
    }
}

impl std::ops::MulAssign for SymExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl std::ops::Div for SymExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl std::fmt::Display for SymExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => {
                if value < &0.0 {
                    write!(f, "(- {})", -value)
                } else {
                    write!(f, "{value}")
                }
            }
            Self::Variable(i) => write!(f, "x{i}"),
            Self::Add(lhs, rhs) => write!(f, "(+ {lhs} {rhs})"),
            Self::Mul(lhs, rhs) => {
                if Self::Constant(-1.0) == **rhs {
                    write!(f, "(- {lhs})")
                } else {
                    write!(f, "(* {lhs} {rhs})")
                }
            }
            Self::Pow(expr, n) => {
                if *n == -1 {
                    write!(f, "(/ 1 {expr})")
                } else if *n < 0 {
                    write!(f, "(/ 1 (^ {expr} {}))", -n)
                } else {
                    write!(f, "(^ {expr} {n})")
                }
            }
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

    pub fn to_z3<'a>(&self, ctx: &'a z3::Context) -> z3::ast::Bool<'a> {
        match self {
            SymConstraint::Eq(e1, e2) => z3::ast::Ast::_eq(&e1.to_z3(ctx), &e2.to_z3(ctx)),
            SymConstraint::Lt(e1, e2) => e1.to_z3(ctx).lt(&e2.to_z3(ctx)),
            SymConstraint::Le(e1, e2) => e1.to_z3(ctx).le(&e2.to_z3(ctx)),
            SymConstraint::Or(constraints) => {
                let disjuncts = constraints.iter().map(|c| c.to_z3(ctx)).collect::<Vec<_>>();
                z3::ast::Bool::or(ctx, &disjuncts.iter().collect::<Vec<_>>())
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

    pub fn substitute(&self, replacements: &[SymExpr]) -> SymConstraint {
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

    pub fn extract_linear(&self) -> Option<LinearConstraint> {
        match self {
            SymConstraint::Eq(e1, e2) => Some(LinearConstraint::eq(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Lt(..) => None,
            SymConstraint::Le(e1, e2) => Some(LinearConstraint::le(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Or(constraints) => {
                // Here we only support constraints without variables
                for constraint in constraints {
                    if let Some(linear_constraint) = constraint.extract_linear() {
                        if linear_constraint.eval_constant() == Some(true) {
                            return Some(LinearConstraint::eq(
                                LinearExpr::constant(0.0),
                                LinearExpr::constant(0.0),
                            ));
                        }
                    }
                }
                return None;
            }
        }
    }
}

impl std::fmt::Display for SymConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(e1, e2) => write!(f, "(= {e1} {e2})"),
            Self::Lt(e1, e2) => write!(f, "(< {e1} {e2})"),
            Self::Le(e1, e2) => write!(f, "(<= {e1} {e2})"),
            Self::Or(constraints) => {
                write!(f, "(or")?;
                for constraint in constraints {
                    write!(f, " {}", constraint)?;
                }
                write!(f, ")")
            }
        }
    }
}
