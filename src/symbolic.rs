use std::{cmp::Ordering, fmt::Display, rc::Rc};

use crate::{
    generating_function::factorial_moments_to_moments, number::Number, ppl::Var,
    semantics::support::VarSupport, univariate_taylor::TaylorExpansion,
};

use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use SymGenFunKind::*;

fn same_ref<T: ?Sized>(a: &Rc<T>, b: &Rc<T>) -> bool {
    std::ptr::eq(a.as_ref(), b.as_ref())
}

#[derive(Clone, Debug)]
pub struct SymGenFun<T> {
    root: Rc<SymGenFunKind<T>>,
}

impl<T> SymGenFun<T> {
    fn new(root: Rc<SymGenFunKind<T>>) -> Self {
        Self { root }
    }

    pub fn var(var: Var) -> Self {
        Self::new(SymGenFunKind::var(var))
    }

    pub fn lit(x: T) -> Self {
        Self::new(SymGenFunKind::lit(x))
    }

    pub fn substitute_var(&self, v: Var, val: Self) -> Self
    where
        T: Number,
    {
        let map = |w| if w == v { Some(val.clone().root) } else { None };
        SymGenFun::new(SymGenFunKind::substitute(&self.root, &map))
    }

    pub fn derive(&self, var: Var) -> Self
    where
        T: Number,
    {
        SymGenFun::new(SymGenFunKind::derive(&self.root, var))
    }

    pub fn taylor_coeffs_at(&self, var: Var, x: &T, order: usize) -> TaylorExpansion<Self>
    where
        T: Number,
    {
        let mut cache = FxHashMap::default();
        SymGenFunKind::taylor_coeffs_with(&self.root, var, Some(x), order, &mut cache)
    }

    pub fn taylor_coeffs(&self, var: Var, order: usize) -> TaylorExpansion<Self>
    where
        T: Number,
    {
        let mut cache = FxHashMap::default();
        SymGenFunKind::taylor_coeffs_with(&self.root, var, None, order, &mut cache)
    }

    pub fn evaluate<S: Number>(&self, lit_map: impl Fn(&T) -> S, var_map: impl Fn(Var) -> S) -> S
    where
        T: Number,
    {
        self.root.clone().evaluate(&lit_map, &var_map)
    }

    pub fn evaluate_vars(&self, var_map: impl Fn(Var) -> T) -> T
    where
        T: Number,
    {
        self.evaluate(std::clone::Clone::clone, var_map)
    }

    pub fn evaluate_var<S: From<T> + Number>(&self, v: Var, val: S) -> S
    where
        T: Number,
    {
        self.evaluate(
            |x| x.clone().into(),
            |w| {
                assert_eq!(w, v, "only expected variable {v}, but found {w}");
                val.clone()
            },
        )
    }

    pub fn evaluate_closed(&self) -> T
    where
        T: Number,
    {
        self.evaluate(std::clone::Clone::clone, |_| {
            unreachable!("term should be closed")
        })
    }
}

impl<T: Number> Zero for SymGenFun<T> {
    fn zero() -> Self {
        Self::new(SymGenFunKind::zero())
    }

    fn is_zero(&self) -> bool {
        match self.root.as_ref() {
            SymGenFunKind::Lit(x) => x.is_zero(),
            _ => false,
        }
    }
}

impl<T: Number> One for SymGenFun<T> {
    fn one() -> Self {
        Self::new(SymGenFunKind::one())
    }

    fn is_one(&self) -> bool {
        match self.root.as_ref() {
            SymGenFunKind::Lit(x) => x.is_one(),
            _ => false,
        }
    }
}

impl<T: Number> From<u32> for SymGenFun<T> {
    fn from(x: u32) -> Self {
        Self::new(SymGenFunKind::lit(x.into()))
    }
}

impl<T: Number> PartialEq for SymGenFun<T> {
    fn eq(&self, other: &Self) -> bool {
        same_ref(&self.root, &other.root)
    }
}

impl<T: Number> PartialOrd for SymGenFun<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
}

impl<T: Number> Number for SymGenFun<T> {
    fn exp(&self) -> Self {
        SymGenFun::new(SymGenFunKind::exp(self.root.clone()))
    }

    fn log(&self) -> Self {
        SymGenFun::new(SymGenFunKind::log(self.root.clone()))
    }

    fn pow(&self, exp: u32) -> Self {
        SymGenFun::new(SymGenFunKind::pow(self.root.clone(), exp))
    }

    fn max(&self, other: &Self) -> Self {
        SymGenFun::new(SymGenFunKind::max(self.root.clone(), other.root.clone()))
    }
}

impl<T: Number + Display> Display for SymGenFun<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.root.fmt(f)
    }
}

impl<T: Number> std::ops::Add for SymGenFun<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SymGenFun::new(SymGenFunKind::add(self.root, rhs.root))
    }
}

impl<T: Number> std::ops::AddAssign for SymGenFun<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.root = SymGenFunKind::add(self.root.clone(), rhs.root);
    }
}

impl<T: Number> std::ops::Neg for SymGenFun<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        SymGenFun::new(SymGenFunKind::neg(self.root))
    }
}

impl<T: Number> std::ops::Sub for SymGenFun<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: Number> std::ops::SubAssign for SymGenFun<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<T: Number> std::ops::Mul for SymGenFun<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SymGenFun::new(SymGenFunKind::mul(self.root, rhs.root))
    }
}

impl<T: Number> std::ops::MulAssign for SymGenFun<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: Number> std::ops::Div for SymGenFun<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        SymGenFun::new(SymGenFunKind::div(self.root, rhs.root))
    }
}

impl<T: Number> std::ops::DivAssign for SymGenFun<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

pub fn probs_symbolic<T: Number>(
    pgf: &SymGenFun<T>,
    v: Var,
    var_info: &VarSupport,
    n: usize,
) -> Vec<T> {
    let var = TaylorExpansion::var(T::zero(), n);
    let taylor = pgf.evaluate(
        |x| TaylorExpansion::Constant(x.clone()),
        |w| {
            if w == v {
                var.clone()
            } else if var_info[w].is_discrete() {
                TaylorExpansion::one()
            } else {
                TaylorExpansion::zero()
            }
        },
    );
    (0..n).map(|i| taylor.coeff(i)).collect()
}

pub fn moments_symbolic<T: Number>(
    pgf: &SymGenFun<T>,
    v: Var,
    var_info: &VarSupport,
    limit: usize,
) -> (T, Vec<T>) {
    let var = if var_info[v].is_discrete() {
        TaylorExpansion::var(T::one(), limit)
    } else {
        TaylorExpansion::var(T::zero(), limit)
    };
    let taylor = pgf.evaluate(
        |x| TaylorExpansion::Constant(x.clone()),
        |w| {
            if w == v {
                var.clone()
            } else if var_info[w].is_discrete() {
                TaylorExpansion::one()
            } else {
                TaylorExpansion::zero()
            }
        },
    );
    let mut result = Vec::with_capacity(limit);
    let mut factor = T::one();
    for i in 0..limit {
        result.push(taylor.coeff(i) * factor.clone());
        factor *= T::from((i + 1) as u32);
    }
    if var_info[v].is_discrete() {
        factorial_moments_to_moments(&result)
    } else {
        let total = result[0].clone();
        let moments = result[1..]
            .iter()
            .map(|x| x.clone() / total.clone())
            .collect();
        (total, moments)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum SymGenFunKind<T> {
    Variable(Var),
    Lit(T),
    Add(Rc<SymGenFunKind<T>>, Rc<SymGenFunKind<T>>),
    Mul(Rc<SymGenFunKind<T>>, Rc<SymGenFunKind<T>>),
    Div(Rc<SymGenFunKind<T>>, Rc<SymGenFunKind<T>>),
    Exp(Rc<SymGenFunKind<T>>),
    Log(Rc<SymGenFunKind<T>>),
    Pow(Rc<SymGenFunKind<T>>, u32),
    Max(Rc<SymGenFunKind<T>>, Rc<SymGenFunKind<T>>),
}

impl<T> SymGenFunKind<T> {
    fn precedence(&self) -> usize {
        match self {
            Variable(_) | Lit(_) | Exp(_) | Log(_) | Max(..) => 10,
            Add(_, _) => 0,
            Mul(_, _) | Div(_, _) => 1,
            Pow(_, _) => 2,
        }
    }

    pub fn evaluate<S: Number>(
        self: &Rc<SymGenFunKind<T>>,
        lit_map: &impl Fn(&T) -> S,
        var_map: &impl Fn(Var) -> S,
    ) -> S
    where
        T: Number,
    {
        let mut cache = FxHashMap::default();
        self.evaluate_with(lit_map, var_map, &mut cache)
    }

    fn evaluate_with<S: Number>(
        self: &Rc<SymGenFunKind<T>>,
        lit_map: &impl Fn(&T) -> S,
        var_map: &impl Fn(Var) -> S,
        cache: &mut FxHashMap<usize, (Rc<SymGenFunKind<T>>, S)>,
    ) -> S
    where
        T: Number,
    {
        let key = self.as_ref() as *const SymGenFunKind<T> as usize;
        if let Some((_, result)) = cache.get(&key) {
            return result.clone();
        }
        let result = match self.as_ref() {
            Variable(v) => var_map(*v),
            Lit(l) => lit_map(l),
            Add(lhs, rhs) => {
                lhs.clone().evaluate_with(lit_map, var_map, cache)
                    + rhs.clone().evaluate_with(lit_map, var_map, cache)
            }
            Mul(lhs, rhs) => {
                lhs.clone().evaluate_with(lit_map, var_map, cache)
                    * rhs.clone().evaluate_with(lit_map, var_map, cache)
            }
            Div(lhs, rhs) => {
                lhs.clone().evaluate_with(lit_map, var_map, cache)
                    / rhs.clone().evaluate_with(lit_map, var_map, cache)
            }
            Exp(arg) => arg.evaluate_with(lit_map, var_map, cache).exp(),
            Log(arg) => arg.evaluate_with(lit_map, var_map, cache).log(),
            Pow(base, exp) => base.evaluate_with(lit_map, var_map, cache).pow(*exp),
            Max(lhs, rhs) => lhs
                .evaluate_with(lit_map, var_map, cache)
                .max(&rhs.clone().evaluate_with(lit_map, var_map, cache)),
        };
        cache.insert(key, (self.clone(), result.clone()));
        result
    }

    fn zero() -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        Self::lit(T::zero())
    }

    fn one() -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        Self::lit(T::one())
    }

    fn lit(x: T) -> Rc<SymGenFunKind<T>> {
        Rc::new(SymGenFunKind::Lit(x))
    }

    fn var(var: Var) -> Rc<SymGenFunKind<T>> {
        Rc::new(SymGenFunKind::Variable(var))
    }

    fn add(lhs: Rc<SymGenFunKind<T>>, rhs: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        match (lhs.as_ref(), rhs.as_ref()) {
            (Lit(a), _) if a.is_zero() => return rhs,
            (_, Lit(b)) if b.is_zero() => return lhs,
            (Lit(a), Lit(b)) => return Self::lit(a.clone() + b.clone()),
            (Lit(l), Add(b, a)) => match a.as_ref() {
                Lit(a) => return Self::add(b.clone(), Self::lit(l.clone() + a.clone())),
                _ => return Rc::new(Add(rhs, lhs)),
            },
            (Add(a, b), Lit(l)) => match a.as_ref() {
                Lit(a) => return Self::add(b.clone(), Self::lit(l.clone() + a.clone())),
                _ => return Rc::new(Add(lhs, rhs)),
            },
            (Add(a, b), Add(c, d)) => match (b.as_ref(), d.as_ref()) {
                (Lit(b), Lit(d)) => {
                    return Self::add(
                        Self::add(a.clone(), c.clone()),
                        Self::lit(b.clone() + d.clone()),
                    )
                }
                (Lit(_), _) => return Self::add(Self::add(a.clone(), rhs), b.clone()),
                (_, Lit(_)) => return Self::add(Self::add(lhs, c.clone()), d.clone()),
                _ => {}
            },
            _ => {}
        }
        Rc::new(Add(lhs, rhs))
    }

    fn mul(lhs: Rc<SymGenFunKind<T>>, rhs: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        // literal simplifications:
        match (lhs.as_ref(), rhs.as_ref()) {
            (Lit(a), _) if a.is_zero() => return Self::zero(),
            (_, Lit(b)) if b.is_zero() => return Self::zero(),
            (Lit(a), _) if a.is_one() => return rhs,
            (_, Lit(b)) if b.is_one() => return lhs,
            (Exp(a), Exp(b)) => return Self::exp(Self::add(a.clone(), b.clone())),
            (Lit(a), Lit(b)) => return Self::lit(a.clone() * b.clone()),
            (Lit(a), Mul(b1, b2)) | (Mul(b1, b2), Lit(a)) => {
                if let Lit(b1) = b1.as_ref() {
                    return Rc::new(Mul(Self::lit(a.clone() * b1.clone()), b2.clone()));
                }
            }
            _ => {}
        }
        // exp simplifications:
        match (lhs.as_ref(), rhs.as_ref()) {
            (Mul(a1, a2), Exp(b)) | (Exp(b), Mul(a1, a2)) => {
                if let Exp(a2) = a2.as_ref() {
                    return Self::mul(a1.clone(), Self::exp(Self::add(a2.clone(), b.clone())));
                }
                if let Exp(a1) = a1.as_ref() {
                    return Self::mul(a2.clone(), Self::exp(Self::add(a1.clone(), b.clone())));
                }
            }
            (Mul(a1, a2), Mul(b1, b2)) => {
                match (a1.as_ref(), a2.as_ref(), b1.as_ref(), b2.as_ref()) {
                    (Exp(a1), _, Exp(b1), _) => {
                        return Self::mul(
                            Self::mul(a2.clone(), b2.clone()),
                            Self::exp(Self::add(a1.clone(), b1.clone())),
                        );
                    }
                    (Exp(a1), _, _, Exp(b2)) => {
                        return Self::mul(
                            Self::mul(a2.clone(), b1.clone()),
                            Self::exp(Self::add(a1.clone(), b2.clone())),
                        );
                    }
                    (_, Exp(a2), Exp(b1), _) => {
                        return Self::mul(
                            Self::mul(a1.clone(), b2.clone()),
                            Self::exp(Self::add(a2.clone(), b1.clone())),
                        );
                    }
                    (_, Exp(a2), _, Exp(b2)) => {
                        return Self::mul(
                            Self::mul(a1.clone(), b1.clone()),
                            Self::exp(Self::add(a2.clone(), b2.clone())),
                        );
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        // moving literals left:
        match (lhs.as_ref(), rhs.as_ref()) {
            (Mul(a1, a2), Mul(b1, b2)) => {
                if let (Lit(a1), Lit(b1)) = (a1.as_ref(), b1.as_ref()) {
                    return Rc::new(Mul(
                        Self::lit(a1.clone() * b1.clone()),
                        Self::mul(a2.clone(), b2.clone()),
                    ));
                }
            }
            (Mul(a1, a2), _) => {
                if let Lit(_) = a1.as_ref() {
                    return Rc::new(Mul(a1.clone(), Self::mul(a2.clone(), rhs)));
                }
            }
            (_, Mul(b1, b2)) => {
                if let Lit(_) = b1.as_ref() {
                    return Rc::new(Mul(b1.clone(), Self::mul(b2.clone(), lhs)));
                }
            }
            _ => {}
        }
        // pow simplifications:
        match (lhs.as_ref(), rhs.as_ref()) {
            (Mul(a1, a2), _) if same_ref(a2, &rhs) => {
                return Self::mul(a1.clone(), Self::pow(a2.clone(), 2))
            }
            (Mul(a1, a2), Pow(b, n)) if same_ref(a2, b) => {
                return Self::mul(a1.clone(), Self::pow(a2.clone(), n + 1))
            }
            (Mul(a1, a2), Pow(b, exp2)) => {
                if let Pow(a21, exp1) = a2.as_ref() {
                    if same_ref(a21, b) {
                        return Self::mul(a1.clone(), Self::pow(a21.clone(), exp1 + exp2));
                    }
                }
            }
            _ => {}
        }
        if let Lit(_) = rhs.as_ref() {
            return Rc::new(Mul(rhs, lhs));
        }
        Rc::new(Mul(lhs, rhs))
    }

    fn div(lhs: Rc<SymGenFunKind<T>>, rhs: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        match (lhs.as_ref(), rhs.as_ref()) {
            (Lit(a), _) if a.is_zero() => Self::zero(),
            (_, Lit(b)) if b.is_one() => lhs,
            _ => Rc::new(Div(lhs, rhs)),
        }
    }

    fn neg(arg: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        SymGenFunKind::mul(Self::lit(-T::one()), arg)
    }

    fn exp(arg: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        match &*arg {
            Lit(a) if a.is_zero() => return Self::one(),
            Lit(a) => return Self::lit(a.exp()),
            Add(a, b) => {
                if let Lit(b) = b.as_ref() {
                    return Self::mul(Self::lit(b.exp()), Self::exp(a.clone()));
                }
            }
            _ => {}
        }
        Rc::new(Exp(arg))
    }

    fn log(arg: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        match &*arg {
            Lit(a) if a.is_one() => return Self::zero(),
            Lit(a) => return Self::lit(a.log()),
            Mul(a, b) => {
                if let Lit(a) = a.as_ref() {
                    return Self::add(Self::log(b.clone()), Self::lit(a.log()));
                }
            }
            _ => {}
        }
        Rc::new(Log(arg))
    }

    fn pow(base: Rc<SymGenFunKind<T>>, exp: u32) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        if exp == 0 {
            Self::one()
        } else if exp == 1 {
            base
        } else {
            match &*base {
                Lit(a) if a.is_zero() => Self::zero(),
                Lit(a) if a.is_one() => Self::one(),
                // Lit(a) => Rc::new(Lit(a.pow(exp))),
                _ => Rc::new(Pow(base, exp)),
            }
        }
    }

    fn max(lhs: Rc<SymGenFunKind<T>>, rhs: Rc<SymGenFunKind<T>>) -> Rc<SymGenFunKind<T>> {
        Rc::new(Max(lhs, rhs))
    }

    fn substitute(
        term: &Rc<SymGenFunKind<T>>,
        map: &impl Fn(Var) -> Option<Rc<SymGenFunKind<T>>>,
    ) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        let mut cache = FxHashMap::default();
        Self::substitute_with(term, map, &mut cache)
    }

    fn substitute_with(
        term: &Rc<SymGenFunKind<T>>,
        map: &impl Fn(Var) -> Option<Rc<SymGenFunKind<T>>>,
        cache: &mut FxHashMap<usize, Rc<SymGenFunKind<T>>>,
    ) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        let key = term.as_ref() as *const SymGenFunKind<T> as usize;
        if let Some(result) = cache.get(&key) {
            return result.clone();
        }
        let result = match term.as_ref() {
            Variable(v) => {
                if let Some(val) = map(*v) {
                    val
                } else {
                    term.clone()
                }
            }
            Lit(_) => term.clone(),
            Add(a, b) => {
                let a2 = Self::substitute_with(a, map, cache);
                let b2 = Self::substitute_with(b, map, cache);
                if same_ref(a, &a2) && same_ref(b, &b2) {
                    term.clone()
                } else {
                    Self::add(a2, b2)
                }
            }
            Mul(a, b) => {
                let a2 = Self::substitute_with(a, map, cache);
                let b2 = Self::substitute_with(b, map, cache);
                if same_ref(a, &a2) && same_ref(b, &b2) {
                    term.clone()
                } else {
                    Self::mul(a2, b2)
                }
            }
            Div(a, b) => {
                let a2 = Self::substitute_with(a, map, cache);
                let b2 = Self::substitute_with(b, map, cache);
                if same_ref(a, &a2) && same_ref(b, &b2) {
                    term.clone()
                } else {
                    Self::div(a2, b2)
                }
            }
            Exp(a) => {
                let a2 = Self::substitute_with(a, map, cache);
                if same_ref(a, &a2) {
                    term.clone()
                } else {
                    Self::exp(a2)
                }
            }
            Log(a) => {
                let a2 = Self::substitute_with(a, map, cache);
                if same_ref(a, &a2) {
                    term.clone()
                } else {
                    Self::log(a2)
                }
            }
            Pow(a, exp) => {
                let a2 = Self::substitute_with(a, map, cache);
                if same_ref(a, &a2) {
                    term.clone()
                } else {
                    Self::pow(a2, *exp)
                }
            }
            Max(a, b) => {
                let a2 = Self::substitute_with(a, map, cache);
                let b2 = Self::substitute_with(b, map, cache);
                if same_ref(a, &a2) && same_ref(b, &b2) {
                    term.clone()
                } else {
                    Self::max(a2, b2)
                }
            }
        };
        // Only cache the result if it is shared (speed-up by almost an order of magnitude in some cases)
        if Rc::strong_count(term) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }

    fn derive(term: &Rc<SymGenFunKind<T>>, var: Var) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        let mut cache = FxHashMap::default();
        Self::derive_with(term, var, &mut cache)
    }

    fn derive_with(
        term: &Rc<SymGenFunKind<T>>,
        var: Var,
        cache: &mut FxHashMap<usize, Rc<SymGenFunKind<T>>>,
    ) -> Rc<SymGenFunKind<T>>
    where
        T: Number,
    {
        let key = term.as_ref() as *const SymGenFunKind<T> as usize;
        if let Some(result) = cache.get(&key) {
            return result.clone();
        }
        let result = match term.as_ref() {
            Variable(v) => {
                if *v == var {
                    Self::one()
                } else {
                    Self::zero()
                }
            }
            Lit(_) => Self::zero(),
            Add(a, b) => {
                let da = Self::derive_with(a, var, cache);
                let db = Self::derive_with(b, var, cache);
                Self::add(da, db)
            }
            Mul(a, b) => {
                let da = Self::derive_with(a, var, cache);
                let db = Self::derive_with(b, var, cache);
                let x = Self::mul(a.clone(), db);
                let y = Self::mul(b.clone(), da);
                Self::add(x, y)
            }
            Div(a, b) => {
                let da = Self::derive_with(a, var, cache);
                let db = Self::derive_with(b, var, cache);
                let x = Self::mul(a.clone(), db);
                let y = Self::mul(b.clone(), da);
                let b2 = Self::pow(b.clone(), 2);
                Self::div(Self::add(x, Self::neg(y)), b2)
            }
            Exp(a) => {
                let da = Self::derive_with(a, var, cache);
                Self::mul(da, term.clone())
            }
            Log(a) => {
                let da = Self::derive_with(a, var, cache);
                Self::div(da, a.clone())
            }
            Pow(a, exp) => {
                assert_ne!(
                    *exp, 0,
                    "unexpected 0 exponent, should have been simplified away before"
                );
                let da = Self::derive_with(a, var, cache);
                let a_exp_minus_1 = Self::pow(a.clone(), exp - 1);
                let tmp = Self::lit(T::from(*exp));
                let tmp = Self::mul(tmp, da);
                Self::mul(tmp, a_exp_minus_1)
            }
            Max(..) => {
                panic!("Maximum shouldn't be differentiated.")
            }
        };
        // Only cache the result if it is shared
        if Rc::strong_count(term) > 1 {
            cache.insert(key, result.clone());
        }
        cache.insert(key, result.clone());
        result
    }

    fn taylor_coeffs_with(
        term: &Rc<SymGenFunKind<T>>,
        var: Var,
        x: Option<&T>,
        order: usize,
        cache: &mut FxHashMap<usize, TaylorExpansion<SymGenFun<T>>>,
    ) -> TaylorExpansion<SymGenFun<T>>
    where
        T: Number,
    {
        let key = term.as_ref() as *const SymGenFunKind<T> as usize;
        if let Some(coeffs) = cache.get(&key) {
            return coeffs.clone();
        }
        let result = match term.as_ref() {
            Variable(v) => {
                if *v == var {
                    let point = if let Some(x) = x {
                        SymGenFun::new(Self::lit(x.clone()))
                    } else {
                        SymGenFun::new(Self::var(var))
                    };
                    TaylorExpansion::var(point, order)
                } else {
                    TaylorExpansion::Constant(SymGenFun::new(term.clone()))
                }
            }
            Lit(_) => TaylorExpansion::Constant(SymGenFun::new(term.clone())),
            Add(a, b) => {
                Self::taylor_coeffs_with(a, var, x, order, cache)
                    + Self::taylor_coeffs_with(b, var, x, order, cache)
            }
            Mul(a, b) => {
                Self::taylor_coeffs_with(a, var, x, order, cache)
                    * Self::taylor_coeffs_with(b, var, x, order, cache)
            }
            Div(a, b) => {
                Self::taylor_coeffs_with(a, var, x, order, cache)
                    / Self::taylor_coeffs_with(b, var, x, order, cache)
            }
            Exp(a) => Self::taylor_coeffs_with(a, var, x, order, cache).exp(),
            Log(a) => Self::taylor_coeffs_with(a, var, x, order, cache).log(),
            Pow(a, exp) => Self::taylor_coeffs_with(a, var, x, order, cache).pow(*exp),
            Max(..) => {
                panic!("Maximum shouldn't be differentiated.")
            }
        };
        // Only cache the result if it is shared (speed-up by almost an order of magnitude in some cases)
        if Rc::strong_count(term) > 1 {
            cache.insert(key, result.clone());
        }
        cache.insert(key, result.clone());
        result
    }
}

fn node_fmt<T: Number + Display>(
    node: &SymGenFunKind<T>,
    parent_precedence: usize,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let cur_precedence = node.precedence();
    if cur_precedence < parent_precedence {
        write!(f, "(")?;
    }
    match node {
        Variable(v) => {
            write!(f, "{v}")?;
        }
        Lit(x) => {
            write!(f, "{x}")?;
        }
        Add(a, b) => {
            node_fmt(a, cur_precedence, f)?;
            write!(f, " + ")?;
            node_fmt(b, cur_precedence, f)?;
        }
        Mul(a, b) => {
            node_fmt(a, cur_precedence, f)?;
            write!(f, "*")?;
            node_fmt(b, cur_precedence, f)?;
        }
        Div(a, b) => {
            node_fmt(a, cur_precedence, f)?;
            write!(f, "/")?;
            node_fmt(b, cur_precedence + 1, f)?;
        }
        Exp(a) => {
            write!(f, "exp(")?;
            node_fmt(a, 0, f)?;
            write!(f, ")")?;
        }
        Log(a) => {
            write!(f, "log(")?;
            node_fmt(a, 0, f)?;
            write!(f, ")")?;
        }
        Pow(a, exp) => {
            node_fmt(a, cur_precedence + 1, f)?;
            write!(f, "^{exp}")?;
        }
        Max(..) => {
            panic!("Maximum shouldn't be differentiated.")
        }
    }
    if cur_precedence < parent_precedence {
        write!(f, ")")?;
    }
    Ok(())
}

impl<T: Number + Display> std::fmt::Display for SymGenFunKind<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        node_fmt(self, 0, f)
    }
}
