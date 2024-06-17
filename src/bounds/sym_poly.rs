use std::{
    collections::{
        hash_map::Entry::{Occupied, Vacant},
        BTreeMap, HashMap,
    },
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::{One, Zero};

use crate::bounds::linear::{LinearConstraint, LinearExpr};

use super::{float_rat::FloatRat, util::pow_nonneg};

#[derive(Debug, Clone, PartialEq)]
pub struct SparsePoly<T> {
    monomials: HashMap<BTreeMap<usize, usize>, T>,
}

impl<T> SparsePoly<T> {
    fn normalize(&mut self)
    where
        T: Zero,
    {
        self.monomials.retain(|_, coeff| !coeff.is_zero());
    }

    pub fn constant(value: T) -> Self {
        Self {
            monomials: HashMap::from([(BTreeMap::new(), value)]),
        }
    }

    pub fn var(i: usize) -> Self
    where
        Self: Zero,
        T: Zero,
        T: One,
    {
        Self {
            monomials: HashMap::from([(BTreeMap::from([(i, 1)]), T::one())]),
        }
    }

    pub fn monomial(monomial: BTreeMap<usize, usize>, coeff: T) -> Self
    where
        Self: Zero,
        T: Zero,
    {
        if coeff.is_zero() {
            Self::zero()
        } else {
            Self {
                monomials: HashMap::from([(monomial, coeff)]),
            }
        }
    }

    pub fn pow(self, mut n: u32) -> Self
    where
        T: Clone + PartialEq,
        Self: Zero + One + MulAssign,
    {
        if n == 0 {
            Self::one()
        } else if n == 1 || self.is_zero() || self.is_one() {
            self
        } else {
            let mut result = Self::one();
            let mut base = self;
            while n > 0 {
                if n % 2 == 1 {
                    result *= base.clone();
                }
                base *= base.clone();
                n /= 2;
            }
            result
        }
    }

    pub fn extract_constant(&self) -> Option<&T> {
        if self.monomials.len() == 1 {
            let (monomial, coeff) = self.monomials.iter().next().unwrap();
            if monomial.is_empty() {
                return Some(coeff);
            }
        }
        None
    }

    pub fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv_to_z3: &impl Fn(&'a z3::Context, &T) -> z3::ast::Real<'a>,
    ) -> z3::ast::Real<'a> {
        let mut result = z3::ast::Real::from_real(ctx, 0, 1);
        for (monomial, coeff) in &self.monomials {
            let mut term = conv_to_z3(ctx, coeff);
            for (var, exp) in monomial {
                let var = z3::ast::Real::new_const(ctx, *var as u32);
                term *= var.power(&z3::ast::Int::from_i64(ctx, *exp as i64).to_real());
            }
            result += term;
        }
        result
    }

    pub fn to_python(&self) -> String
    where
        T: Display,
    {
        let mut result = String::new();
        for (monomial, coeff) in &self.monomials {
            if !result.is_empty() {
                result += " + ";
            }
            result += &coeff.to_string();
            for (var, exp) in monomial {
                result += &format!(" * x[{var}]**{exp}");
            }
        }
        result
    }

    pub fn to_python_z3(&self) -> String
    where
        T: Display,
    {
        let mut result = String::new();
        for (monomial, coeff) in &self.monomials {
            if !result.is_empty() {
                result += " + ";
            }
            result += &coeff.to_string();
            for (var, exp) in monomial {
                result += &format!(" * x{var}^{exp}");
            }
        }
        result
    }

    pub fn to_qepcad(&self) -> String
    where
        T: Display,
    {
        let mut result = String::new();
        for (monomial, coeff) in &self.monomials {
            if !result.is_empty() {
                result += " + ";
            }
            result += &coeff.to_string();
            for (var, exp) in monomial {
                result += &format!(" x{var}^{exp}");
            }
        }
        result
    }

    pub fn eval(&self, values: &[T]) -> T
    where
        T: Clone + Zero + One + AddAssign + MulAssign,
    {
        let mut result = T::zero();
        for (monomial, coeff) in &self.monomials {
            let mut term = coeff.clone();
            for (var, exp) in monomial {
                term *= pow_nonneg(values[*var].clone(), *exp as u32);
            }
            result += term;
        }
        result
    }

    pub fn substitute(&self, replacements: &[Self]) -> Self
    where
        T: Clone + Zero + One + PartialEq + AddAssign + MulAssign,
    {
        let mut result = Self::zero();
        for (monomial, coeff) in &self.monomials {
            let mut term = Self::constant(coeff.clone());
            for (var, exp) in monomial {
                term *= replacements[*var].clone().pow(*exp as u32);
            }
            result += term;
        }
        result
    }

    pub fn derive(&self, var: usize) -> Self
    where
        T: From<u32> + Clone + Zero + One + PartialEq + AddAssign + MulAssign,
    {
        let mut result = Self::zero();
        for (monomial, coeff) in &self.monomials {
            if let Some(exp) = monomial.get(&var) {
                let mut new_monomial = monomial.clone();
                if *exp > 1 {
                    new_monomial.insert(var, exp - 1);
                } else {
                    new_monomial.remove(&var);
                }
                let new_coeff = coeff.clone() * T::from(*exp as u32);
                result.monomials.insert(new_monomial, new_coeff);
            }
        }
        result
    }

    pub fn gradient(&self, vars: &[usize]) -> Vec<Self>
    where
        T: From<u32> + Clone + Zero + One + PartialEq + AddAssign + MulAssign,
    {
        vars.iter().map(|&var| self.derive(var)).collect()
    }
}

impl SparsePoly<FloatRat> {
    pub fn extract_linear(&self) -> Option<LinearExpr> {
        let mut coeffs = Vec::new();
        let mut constant = FloatRat::zero();
        for (monomial, coeff) in &self.monomials {
            if monomial.is_empty() {
                constant = coeff.clone();
                continue;
            }
            if monomial.len() == 1 {
                let (var, exp) = monomial.iter().next().unwrap();
                if *exp == 1 {
                    if coeffs.len() <= *var + 1 {
                        coeffs.extend((coeffs.len()..=*var).map(|_| FloatRat::zero()));
                    }
                    coeffs[*var] = coeff.clone();
                    continue;
                }
            }
            return None;
        }
        Some(LinearExpr::new(coeffs, constant))
    }
}

impl<T: From<f64>> From<f64> for SparsePoly<T> {
    fn from(value: f64) -> Self {
        Self::constant(value.into())
    }
}

impl<T: Zero + Clone + Add<T, Output = T> + AddAssign> Zero for SparsePoly<T> {
    fn zero() -> Self {
        Self {
            monomials: HashMap::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.monomials.is_empty()
    }
}

impl<T> One for SparsePoly<T>
where
    T: Zero + One + Clone + AddAssign + PartialEq,
{
    fn one() -> Self {
        Self::monomial(BTreeMap::new(), T::one())
    }

    fn is_one(&self) -> bool {
        if let Some(coeff) = self.extract_constant() {
            coeff.is_one()
        } else {
            false
        }
    }
}

impl<T> Neg for SparsePoly<T>
where
    T: Clone + Neg<Output = T>,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for coeff in &mut self.monomials.values_mut() {
            *coeff = coeff.clone().neg();
        }
        self
    }
}

impl<T> Add for SparsePoly<T>
where
    Self: AddAssign,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T> AddAssign for SparsePoly<T>
where
    T: AddAssign + Zero,
{
    fn add_assign(&mut self, rhs: Self) {
        for (monomial, coeff) in rhs.monomials {
            match self.monomials.entry(monomial) {
                Occupied(mut entry) => {
                    entry.get_mut().add_assign(coeff);
                }
                Vacant(entry) => {
                    entry.insert(coeff);
                }
            }
        }
        self.normalize();
    }
}

impl<T> Sub for SparsePoly<T>
where
    Self: SubAssign,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<T> SubAssign for SparsePoly<T>
where
    T: Zero + Neg<Output = T> + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (monomial, coeff) in rhs.monomials {
            match self.monomials.entry(monomial) {
                Occupied(mut entry) => {
                    entry.get_mut().sub_assign(coeff);
                }
                Vacant(entry) => {
                    entry.insert(-coeff);
                }
            }
        }
        self.normalize();
    }
}

impl<T> Mul for SparsePoly<T>
where
    Self: MulAssign,
{
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<T> MulAssign for SparsePoly<T>
where
    T: PartialEq + Mul<Output = T> + Clone + Zero + AddAssign,
    Self: Zero + One,
{
    fn mul_assign(&mut self, rhs: Self) {
        if self.is_zero() || rhs.is_one() {
            // Result is unchanged
        } else if self.is_one() {
            *self = rhs;
        } else {
            let mut result = HashMap::new();
            for (monomial1, coeff1) in &self.monomials {
                for (monomial2, coeff2) in &rhs.monomials {
                    let mut monomial = monomial1.clone();
                    for (var, exp) in monomial2 {
                        monomial
                            .entry(*var)
                            .and_modify(|e| *e += exp)
                            .or_insert(*exp);
                    }
                    let coeff = coeff1.clone() * coeff2.clone();
                    result
                        .entry(monomial)
                        .and_modify(|c| *c += coeff.clone())
                        .or_insert(coeff);
                }
            }
            self.monomials = result;
            self.normalize();
        }
    }
}

impl<T> Display for SparsePoly<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (monomial, coeff) in &self.monomials {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{coeff}")?;
            for (var, exp) in monomial {
                write!(f, " x{var}")?;
                if exp > &1 {
                    write!(f, "^{exp}")?;
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum PolyConstraint<T> {
    Eq(SparsePoly<T>, SparsePoly<T>),
    Lt(SparsePoly<T>, SparsePoly<T>),
    Le(SparsePoly<T>, SparsePoly<T>),
    Or(Vec<PolyConstraint<T>>),
}
impl<T> PolyConstraint<T> {
    pub fn or(constraints: Vec<PolyConstraint<T>>) -> Self {
        Self::Or(constraints)
    }

    pub fn holds(&self, values: &[T]) -> bool
    where
        T: Clone + Zero + One + PartialEq + AddAssign + MulAssign + PartialOrd,
    {
        match self {
            PolyConstraint::Eq(lhs, rhs) => lhs.eval(values) == rhs.eval(values),
            PolyConstraint::Lt(lhs, rhs) => lhs.eval(values) < rhs.eval(values),
            PolyConstraint::Le(lhs, rhs) => lhs.eval(values) <= rhs.eval(values),
            PolyConstraint::Or(constraints) => constraints.iter().any(|c| c.holds(values)),
        }
    }

    pub fn has_slack(&self, values: &[T], epsilon: T) -> bool
    where
        T: Clone + Zero + One + PartialEq + AddAssign + MulAssign + PartialOrd,
    {
        match self {
            PolyConstraint::Eq(..) => false,
            PolyConstraint::Lt(lhs, rhs) => lhs.eval(values) + epsilon < rhs.eval(values),
            PolyConstraint::Le(lhs, rhs) => lhs.eval(values) + epsilon <= rhs.eval(values),
            PolyConstraint::Or(constraints) => constraints
                .iter()
                .all(|c| c.has_slack(values, epsilon.clone())),
        }
    }

    pub fn to_z3<'a>(
        &self,
        ctx: &'a z3::Context,
        conv: &impl Fn(&'a z3::Context, &T) -> z3::ast::Real<'a>,
    ) -> z3::ast::Bool<'a> {
        match self {
            PolyConstraint::Eq(e1, e2) => {
                z3::ast::Ast::_eq(&e1.to_z3(ctx, conv), &e2.to_z3(ctx, conv))
            }
            PolyConstraint::Lt(e1, e2) => e1.to_z3(ctx, conv).lt(&e2.to_z3(ctx, conv)),
            PolyConstraint::Le(e1, e2) => e1.to_z3(ctx, conv).le(&e2.to_z3(ctx, conv)),
            PolyConstraint::Or(constraints) => {
                let disjuncts = constraints
                    .iter()
                    .map(|c| c.to_z3(ctx, conv))
                    .collect::<Vec<_>>();
                z3::ast::Bool::or(ctx, &disjuncts.iter().collect::<Vec<_>>())
            }
        }
    }

    pub fn to_qepcad(&self) -> String
    where
        T: Display,
    {
        match self {
            PolyConstraint::Eq(lhs, rhs) => {
                format!("{} = {}", lhs.to_qepcad(), rhs.to_qepcad())
            }
            PolyConstraint::Lt(lhs, rhs) => {
                format!("{} < {}", lhs.to_qepcad(), rhs.to_qepcad())
            }
            PolyConstraint::Le(lhs, rhs) => {
                format!("{} <= {}", lhs.to_qepcad(), rhs.to_qepcad())
            }
            PolyConstraint::Or(cs) => {
                let mut res = "[".to_owned();
                let mut first = true;
                for c in cs {
                    if first {
                        first = false;
                    } else {
                        res += r" \/ ";
                    }
                    res += &c.to_qepcad();
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
            PolyConstraint::Eq(lhs, rhs) => {
                format!("{} == {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            PolyConstraint::Lt(lhs, rhs) => {
                format!("{} < {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            PolyConstraint::Le(lhs, rhs) => {
                format!("{} <= {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            PolyConstraint::Or(cs) => {
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

    pub fn substitute(&self, replacements: &[SparsePoly<T>]) -> PolyConstraint<T>
    where
        T: Clone + Zero + One + PartialEq + AddAssign + MulAssign,
    {
        match self {
            PolyConstraint::Eq(e1, e2) => {
                PolyConstraint::Eq(e1.substitute(replacements), e2.substitute(replacements))
            }
            PolyConstraint::Lt(e1, e2) => {
                PolyConstraint::Lt(e1.substitute(replacements), e2.substitute(replacements))
            }
            PolyConstraint::Le(e1, e2) => {
                PolyConstraint::Le(e1.substitute(replacements), e2.substitute(replacements))
            }
            PolyConstraint::Or(constraints) => PolyConstraint::Or(
                constraints
                    .iter()
                    .map(|c| c.substitute(replacements))
                    .collect(),
            ),
        }
    }
}

impl PolyConstraint<FloatRat> {
    pub fn extract_linear(&self) -> Option<LinearConstraint> {
        match self {
            PolyConstraint::Eq(e1, e2) => Some(LinearConstraint::eq(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            PolyConstraint::Lt(..) => None,
            PolyConstraint::Le(e1, e2) => Some(LinearConstraint::le(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            PolyConstraint::Or(constraints) => {
                // Here we only support constraints without variables
                for constraint in constraints {
                    if let Some(linear_constraint) = constraint.extract_linear() {
                        if linear_constraint.eval_constant() == Some(true) {
                            return Some(LinearConstraint::eq(
                                LinearExpr::zero(),
                                LinearExpr::zero(),
                            ));
                        }
                    }
                }
                None
            }
        }
    }
}

impl<T> Display for PolyConstraint<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(e1, e2) => write!(f, "{e1} == {e2}"),
            Self::Lt(e1, e2) => write!(f, "{e1} < {e2}"),
            Self::Le(e1, e2) => write!(f, "{e1} <= {e2}"),
            Self::Or(constraints) => {
                write!(f, "(")?;
                let mut first = true;
                for constraint in constraints {
                    if first {
                        first = false;
                    } else {
                        write!(f, " OR ")?;
                    }
                    write!(f, "{constraint}")?;
                }
                write!(f, ")")
            }
        }
    }
}
