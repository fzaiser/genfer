use std::rc::Rc;

use ndarray::Axis;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;

use crate::{
    multivariate_taylor::{fmt_polynomial, TaylorPoly},
    number::{FloatNumber, Number},
    ppl::{PosRatio, Var, VarRange},
    support::SupportSet,
    symbolic::SymGenFun,
};

type Polynomial<T> = ndarray::ArrayD<T>;

#[derive(Clone, Debug, PartialEq)]
pub struct GenFun<T>(Rc<GeneratingFunctionKind<T>>);

impl<T> GenFun<T> {
    fn fmt_prec(&self, parent_prec: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    where
        T: std::fmt::Display + Zero,
    {
        self.0.fmt_prec(parent_prec, f)
    }

    pub fn used_vars(&self) -> VarRange {
        let mut cache = FxHashMap::default();
        self.used_vars_with(&mut cache)
    }

    pub fn used_vars_with(&self, cache: &mut FxHashMap<usize, VarRange>) -> VarRange {
        let key = self.0.as_ref() as *const GeneratingFunctionKind<T> as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return cached_result.clone();
            }
        }
        let result = self.0.used_vars(cache);
        // Only cache the result if it is shared:
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }
}

impl<T: Number> GenFun<T> {
    pub fn var(v: Var) -> GenFun<T> {
        GeneratingFunctionKind::Var(v).into_gf()
    }

    pub fn constant(x: T) -> GenFun<T> {
        GeneratingFunctionKind::Const(x).into_gf()
    }

    pub fn zero() -> GenFun<T> {
        Self::constant(T::zero())
    }

    pub fn one() -> GenFun<T> {
        Self::constant(T::one())
    }

    pub fn from_u32(n: u32) -> GenFun<T> {
        Self::constant(T::from(n))
    }

    pub fn from_ratio(ratio: PosRatio) -> GenFun<T> {
        Self::constant(T::from_ratio(ratio.numer, ratio.denom))
    }

    pub fn polynomial(coeffs: ndarray::ArrayD<T>) -> GenFun<T> {
        GeneratingFunctionKind::Polynomial(Box::new(coeffs)).into_gf()
    }

    pub fn exp(&self) -> GenFun<T> {
        GeneratingFunctionKind::Exp(self.clone()).into_gf()
    }

    pub fn log(&self) -> GenFun<T> {
        GeneratingFunctionKind::Log(self.clone()).into_gf()
    }

    pub fn pow(&self, n: u32) -> GenFun<T> {
        GeneratingFunctionKind::Pow(self.clone(), n).into_gf()
    }

    pub fn uniform_mgf(g: Self) -> GenFun<T> {
        GeneratingFunctionKind::UniformMgf(g).into_gf()
    }

    /// The derivative of `self` with respect to `v` of order `order`.
    pub fn derive(&self, v: Var, order: usize) -> GenFun<T> {
        GeneratingFunctionKind::Derivative(self.clone(), v, order).into_gf()
    }

    /// Taylor polynomial up to ε^`order` at (0 + ε).
    pub fn taylor_polynomial_at_zero(&self, v: Var, orders: Vec<usize>) -> GenFun<T> {
        GeneratingFunctionKind::TaylorPolynomial(self.clone(), v, orders).into_gf()
    }

    ///Taylor polynomial up to ε^`order` at (`x` + ε).
    pub fn taylor_polynomial_at(&self, v: Var, x: T, orders: Vec<usize>) -> GenFun<T> {
        self.taylor_polynomial_at_zero(v, orders)
            .substitute_var(v, GenFun::constant(x))
    }

    /// Coefficient of ε^`order` in the Taylor expansion of `self` wrt `v` at (0 + ε).
    pub fn taylor_coeff_at_zero(&self, v: Var, order: usize) -> GenFun<T> {
        GeneratingFunctionKind::TaylorCoeffAtZero(self.clone(), v, order).into_gf()
    }

    /// Coefficient of ε^`order` in the Taylor expansion of `self` wrt `v` at (`x` + ε).
    pub fn taylor_coeff_at(&self, v: Var, x: T, order: usize) -> GenFun<T> {
        self.taylor_coeff_at_zero(v, order)
            .substitute_var(v, GenFun::constant(x))
    }

    /// Coefficient of ε^`order` in the Taylor expansion of `self` wrt `v` at (`v` + ε).
    pub fn taylor_coeff(&self, v: Var, order: usize) -> GenFun<T> {
        GeneratingFunctionKind::TaylorCoeff(self.clone(), v, order).into_gf()
    }

    pub fn substitute_var(&self, v: Var, val: Self) -> Self {
        GeneratingFunctionKind::Subst(self.clone(), v, val).into_gf()
    }

    pub fn substitute_all(&self, val: Self) -> Self {
        let num_vars = self.used_vars().num_vars();
        let mut result = self.clone();
        for v in 0..num_vars {
            result = result.substitute_var(Var(v), val.clone());
        }
        result
    }

    #[inline]
    pub fn simplify(&self) -> GenFun<T> {
        let mut cache = FxHashMap::default();
        match self.simplify_with(&mut cache) {
            Some(taylor) => GenFun::polynomial(taylor.into_array()),
            None => self.clone(),
        }
    }

    #[inline]
    fn simplify_with(
        &self,
        cache: &mut FxHashMap<usize, Option<TaylorPoly<T>>>,
    ) -> Option<TaylorPoly<T>> {
        let key = self.0.as_ref() as *const GeneratingFunctionKind<T> as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                return cached_result.clone();
            }
        }
        let result = self.0.simplify(cache);
        // Only cache the result if it is shared:
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(key, result.clone());
        }
        result
    }

    #[inline]
    pub fn eval(&self, inputs: &[T], target_total_degree_p1: usize) -> TaylorPoly<T> {
        let mut cache = FxHashMap::default();
        self.eval_with(inputs, target_total_degree_p1, &mut cache)
    }

    #[inline]
    fn eval_with(
        &self,
        inputs: &[T],
        target_total_degree_p1: usize,
        cache: &mut FxHashMap<usize, EvalResult<T>>,
    ) -> TaylorPoly<T> {
        let key = self.0.as_ref() as *const GeneratingFunctionKind<T> as usize;
        if Rc::strong_count(&self.0) > 1 {
            if let Some(cached_result) = cache.get(&key) {
                let cached_key =
                    cached_result.gf.0.as_ref() as *const GeneratingFunctionKind<T> as usize;
                if key == cached_key
                    && inputs == cached_result.inputs
                    && target_total_degree_p1 == cached_result.degree_p1
                // TODO: can this be changed to `<=` and is that faster?
                {
                    return cached_result.output.clone();
                }
            }
        }
        let result = self.0.eval(inputs, target_total_degree_p1, cache);
        debug_assert!(result.shape().iter().all(|&x| x == target_total_degree_p1),
            "Unexpected shape of output Taylor polynomial for target degree {target_total_degree_p1}: {:?}\nGF: {self}", result.shape());
        // Only cache the result if it is shared:
        if Rc::strong_count(&self.0) > 1 {
            cache.insert(
                key,
                EvalResult {
                    gf: self.clone(),
                    inputs: inputs.to_vec(),
                    degree_p1: target_total_degree_p1,
                    output: result.clone(),
                },
            );
        }
        result
    }

    pub fn to_computation(&self) -> SymGenFun<T> {
        self.0.to_computation()
    }
}

impl<T: std::fmt::Display + Zero> std::fmt::Display for GenFun<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Number> std::ops::Add for GenFun<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        GeneratingFunctionKind::Add(self, rhs).into_gf()
    }
}

impl<T: Number> std::ops::AddAssign for GenFun<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<T: Number> std::ops::Neg for GenFun<T> {
    type Output = Self;
    fn neg(self) -> Self {
        GeneratingFunctionKind::Neg(self).into_gf()
    }
}

impl<T: Number> std::ops::Sub for GenFun<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<T: Number> std::ops::SubAssign for GenFun<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<T: Number> std::ops::Mul for GenFun<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        GeneratingFunctionKind::Mul(self, rhs).into_gf()
    }
}

impl<T: Number> std::ops::MulAssign for GenFun<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: Number> std::ops::Div for GenFun<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        GeneratingFunctionKind::Div(self, rhs).into_gf()
    }
}

impl<T: Number> std::ops::DivAssign for GenFun<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

struct EvalResult<T> {
    gf: GenFun<T>,
    inputs: Vec<T>,
    degree_p1: usize,
    output: TaylorPoly<T>,
}

#[derive(Debug, PartialEq)]
enum GeneratingFunctionKind<T> {
    Var(Var),
    Const(T),
    Add(GenFun<T>, GenFun<T>),
    Neg(GenFun<T>),
    Mul(GenFun<T>, GenFun<T>),
    Div(GenFun<T>, GenFun<T>),
    // TODO: use a dedicated polynomial type, sharing code with TaylorPoly.
    Polynomial(Box<Polynomial<T>>),
    Exp(GenFun<T>),
    Log(GenFun<T>),
    Pow(GenFun<T>, u32),
    /// The function (e^x - 1) / x, continuously extended at x = 0.
    UniformMgf(GenFun<T>),
    Subst(GenFun<T>, Var, GenFun<T>),
    Derivative(GenFun<T>, Var, usize),
    TaylorPolynomial(GenFun<T>, Var, Vec<usize>),
    TaylorCoeffAtZero(GenFun<T>, Var, usize),
    TaylorCoeff(GenFun<T>, Var, usize),
}

impl<T> GeneratingFunctionKind<T> {
    fn into_gf(self) -> GenFun<T> {
        GenFun(Rc::new(self))
    }

    fn fmt_prec(&self, parent_prec: usize, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    where
        T: std::fmt::Display + Zero,
    {
        let cur_prec = self.precedence();
        if cur_prec < parent_prec {
            write!(f, "(")?;
        }
        match self {
            Self::Var(v) => write!(f, "{v}")?,
            Self::Const(c) => write!(f, "{c}")?,
            Self::Add(a, b) => {
                a.fmt_prec(cur_prec, f)?;
                write!(f, " + ")?;
                b.fmt_prec(cur_prec, f)?;
            }
            Self::Neg(a) => {
                write!(f, "-")?;
                a.fmt_prec(cur_prec + 1, f)?;
            }
            Self::Mul(a, b) => {
                a.fmt_prec(cur_prec, f)?;
                write!(f, " * ")?;
                b.fmt_prec(cur_prec, f)?;
            }
            Self::Div(a, b) => {
                a.fmt_prec(cur_prec, f)?;
                write!(f, " / ")?;
                b.fmt_prec(cur_prec + 1, f)?;
            }
            Self::Polynomial(coeffs) => {
                fmt_polynomial(f, &coeffs.view())?;
            }
            Self::Exp(a) => {
                write!(f, "exp(")?;
                a.fmt_prec(0, f)?;
                write!(f, ")")?;
            }
            Self::Log(a) => {
                write!(f, "log(")?;
                a.fmt_prec(0, f)?;
                write!(f, ")")?;
            }
            Self::Pow(a, b) => {
                a.fmt_prec(cur_prec + 1, f)?;
                write!(f, "^{b}")?;
            }
            Self::UniformMgf(a) => {
                write!(f, "uniform_mgf(")?;
                a.fmt_prec(0, f)?;
                write!(f, ")")?;
            }
            Self::Subst(g, v, subst) => {
                write!(f, "[{v} -> ")?;
                subst.fmt_prec(0, f)?;
                write!(f, " in ")?;
                g.fmt_prec(0, f)?;
                write!(f, "]")?;
            }
            Self::Derivative(g, v, order) => {
                write!(f, "d_{v}^{order}(")?;
                g.fmt_prec(0, f)?;
                write!(f, ")")?;
            }
            Self::TaylorPolynomial(g, v, order) => {
                write!(f, "taylor(")?;
                g.fmt_prec(0, f)?;
                write!(f, " of {v}^i with i ∈ {order:?})")?;
            }
            Self::TaylorCoeffAtZero(g, v, order) => {
                write!(f, "coeff_at_zero(")?;
                g.fmt_prec(0, f)?;
                write!(f, " of {v}^{order})")?;
            }
            Self::TaylorCoeff(g, v, order) => {
                write!(f, "coeff(")?;
                g.fmt_prec(0, f)?;
                write!(f, " of {v}^{order})")?;
            }
        }
        if cur_prec < parent_prec {
            write!(f, ")")?;
        }
        Ok(())
    }

    fn used_vars(&self, cache: &mut FxHashMap<usize, VarRange>) -> VarRange {
        match self {
            Self::Var(v) => VarRange::new(*v),
            Self::Const(_) => VarRange::empty(),
            Self::Add(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                a.used_vars_with(cache).union(&b.used_vars_with(cache))
            }
            Self::Neg(a) | Self::Exp(a) | Self::Log(a) | Self::Pow(a, _) | Self::UniformMgf(a) => {
                a.used_vars_with(cache)
            }
            Self::Polynomial(coeffs) => VarRange::zero_to(coeffs.ndim()),
            Self::Subst(g, v, subst) => g
                .used_vars_with(cache)
                .remove(*v)
                .union(&subst.used_vars_with(cache)),
            Self::TaylorCoeffAtZero(g, v, _) => g.used_vars_with(cache).remove(*v),
            Self::Derivative(g, _, _)
            | Self::TaylorPolynomial(g, _, _)
            | Self::TaylorCoeff(g, _, _) => g.used_vars_with(cache),
        }
    }

    pub fn precedence(&self) -> usize {
        use GeneratingFunctionKind::*;
        match self {
            Var(_)
            | Const(_)
            | Exp(_)
            | Log(_)
            | UniformMgf(_)
            | Subst(..)
            | Derivative(..)
            | TaylorPolynomial(..)
            | TaylorCoeffAtZero(..)
            | TaylorCoeff(..) => 10,
            Add(..) | Neg(_) | Polynomial(..) => 0,
            Mul(..) | Div(..) => 1,
            Pow(..) => 2,
        }
    }
}

impl<T: Number> GeneratingFunctionKind<T> {
    fn simplify(
        &self,
        cache: &mut FxHashMap<usize, Option<TaylorPoly<T>>>,
    ) -> Option<TaylorPoly<T>> {
        match self {
            Self::Var(v) => {
                let mut shape = vec![1; v.id() + 1];
                shape[v.id()] = 2;
                Some(TaylorPoly::var_with_degrees_p1(
                    *v,
                    T::zero(),
                    vec![usize::MAX; v.id() + 1],
                ))
            }
            Self::Const(x) => Some(TaylorPoly::from(x.clone())),
            Self::Add(g, h) => match (g.simplify_with(cache), h.simplify_with(cache)) {
                (Some(p1), Some(p2)) => Some(p1 + p2),
                _ => None,
            },
            Self::Neg(g) => g.simplify_with(cache).map(|p| -p),
            Self::Mul(g, h) => match (g.simplify_with(cache), h.simplify_with(cache)) {
                (Some(p1), Some(p2)) => Some(p1 * p2),
                _ => None,
            },
            Self::Div(g, h) => match (g.simplify_with(cache), h.simplify_with(cache)) {
                (Some(p1), Some(p2)) => {
                    if p2.extract_constant().is_some() {
                        Some(p1 / p2)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            Self::Polynomial(_) | Self::Exp(_) | Self::Log(_) | Self::UniformMgf(_) => None,
            Self::Pow(g, n) => g.simplify_with(cache).map(|p| p.pow(*n)),
            Self::Subst(g, v, subst) => {
                match (g.simplify_with(cache), subst.simplify_with(cache)) {
                    (Some(p), Some(q)) => Some(p.subst_var(*v, &q)),
                    _ => None,
                }
            }
            Self::Derivative(g, v, order) => {
                g.simplify_with(cache).map(|p| p.derivative(*v, *order))
            }
            Self::TaylorPolynomial(g, v, orders) => g
                .simplify_with(cache)
                .map(|p| p.taylor_polynomial_terms(*v, orders)),
            Self::TaylorCoeffAtZero(g, v, order) => match g.simplify_with(cache) {
                Some(p) => {
                    let res = p.coefficients_of_term(*v, *order);
                    let res = if v.id() + 1 == res.num_vars() {
                        res.remove_last_variable()
                    } else {
                        res
                    };
                    Some(res)
                }
                _ => None,
            },
            Self::TaylorCoeff(g, v, order) => g
                .simplify_with(cache)
                .map(|p| p.taylor_expansion_of_coeff(*v, *order)),
        }
    }

    fn eval(
        &self,
        inputs: &[T],
        degree_p1: usize,
        cache: &mut FxHashMap<usize, EvalResult<T>>,
    ) -> TaylorPoly<T> {
        match self {
            Self::Var(v) => TaylorPoly::var(*v, inputs[v.id()].clone(), degree_p1),
            Self::Const(x) => TaylorPoly::from(x.clone()),
            Self::Add(g, h) => {
                g.eval_with(inputs, degree_p1, cache) + h.eval_with(inputs, degree_p1, cache)
            }
            Self::Neg(g) => -g.eval_with(inputs, degree_p1, cache),
            Self::Mul(g, h) => {
                g.eval_with(inputs, degree_p1, cache) * h.eval_with(inputs, degree_p1, cache)
            }
            Self::Div(g, h) => {
                g.eval_with(inputs, degree_p1, cache) / h.eval_with(inputs, degree_p1, cache)
            }
            Self::Polynomial(coeffs) => {
                let mut taylor =
                    TaylorPoly::new(coeffs.as_ref().clone(), vec![usize::MAX; coeffs.ndim()]);
                for (v, input) in inputs.iter().enumerate() {
                    taylor = taylor
                        .subst_var(Var(v), &TaylorPoly::var(Var(v), input.clone(), degree_p1));
                }
                let ndim = taylor.num_vars();
                if ndim > inputs.len() {
                    assert!(ndim == inputs.len() + 1);
                    // This can happen because of auxiliary temporary variables introduced for events of the form x ~ D.
                    taylor = taylor.remove_last_variable();
                }
                taylor
                    .extend_to_dim(inputs.len(), degree_p1)
                    .truncate_to_degree_p1(degree_p1)
            }
            Self::Exp(g) => g.eval_with(inputs, degree_p1, cache).exp(),
            Self::Log(g) => g.eval_with(inputs, degree_p1, cache).log(),
            Self::Pow(g, n) => g.eval_with(inputs, degree_p1, cache).pow(*n),
            Self::UniformMgf(g) => {
                let x = g.eval_with(inputs, degree_p1, cache);
                if x.constant_term().is_zero() {
                    let y = TaylorPoly::var_at_zero(Var(0), degree_p1 + 1);
                    let numerator = y.exp() - TaylorPoly::one();
                    let mut numerator_array = numerator.into_array();
                    numerator_array.slice_axis_inplace(ndarray::Axis(0), (1..).into()); // divide by y
                    let fraction = TaylorPoly::new(numerator_array, vec![degree_p1]);
                    fraction.subst_var(Var(0), &x)
                } else {
                    let numerator = x.exp() - TaylorPoly::one();
                    (numerator / x).truncate_to_degree_p1(degree_p1)
                }
            }
            Self::Subst(g, v, replacement) => {
                let mut new_inputs = inputs.to_owned();
                let subst = replacement.eval_with(inputs, degree_p1, cache);
                let c = subst.constant_term();
                let subst = subst - TaylorPoly::from(c.clone());
                if v.id() < inputs.len() {
                    new_inputs[v.id()] = c;
                } else {
                    assert!(v.id() == inputs.len());
                    new_inputs.push(c);
                }
                let taylor = g.eval_with(&new_inputs, degree_p1, cache);
                let mut result = taylor.subst_var(*v, &subst);
                if taylor.shape().len() > inputs.len() {
                    assert!(taylor.shape().len() == inputs.len() + 1);
                    result = result.remove_last_variable();
                }
                result
            }
            Self::Derivative(g, v, order) => {
                let taylor = g.eval_with(inputs, degree_p1 + *order, cache);
                let result = taylor.derivative(*v, *order);
                result.truncate_to_degree_p1(degree_p1)
            }
            Self::TaylorPolynomial(g, v, orders) => {
                let mut new_inputs = inputs.to_owned();
                new_inputs[v.id()] = T::zero();
                let max_order = *orders.iter().max().unwrap_or(&0);
                // TODO: can the target degree_p1 be reduced to degree_p1.max(max_order + 1)?
                let taylor = g.eval_with(&new_inputs, degree_p1 + max_order, cache);
                let result = taylor.taylor_polynomial_terms(*v, orders);
                let result =
                    result.subst_var(*v, &TaylorPoly::var(*v, inputs[v.id()].clone(), degree_p1));
                result.truncate_to_degree_p1(degree_p1)
            }
            Self::TaylorCoeffAtZero(g, v, order) => {
                Self::eval_taylor_coeff_at_zero(g, *v, *order, inputs, degree_p1, cache)
            }
            Self::TaylorCoeff(g, v, order) => {
                let taylor = g.eval_with(inputs, degree_p1 + *order, cache);
                let result = taylor.taylor_expansion_of_coeff(*v, *order);
                result.truncate_to_degree_p1(degree_p1)
            }
        }
    }

    fn eval_taylor_coeff_at_zero(
        g: &GenFun<T>,
        v: Var,
        order: usize,
        inputs: &[T],
        degree_p1: usize,
        cache: &mut FxHashMap<usize, EvalResult<T>>,
    ) -> TaylorPoly<T> {
        if let Some((param_var, lambda, inner)) = recognize_discrete_poisson_observation(g, v) {
            // This is an optimized way of evaluating the observation from a compound Poisson distribution.
            // Normally, we would compute the n-th derivative of G(y*e^(λ(x-1))) and evaluate it at x=0.
            // Instead, we can compute D^n(G) where D(G)(y) := λyG'(y), and evaluate it at y=e^(-λ)y.
            // Finally, we have to divide by n! to get the Taylor coefficient,
            // which we integrate into the loop to improve numerical stability.
            let mut gf = inner.clone();
            for k in 1..=order {
                // The division by `k` is for the factorial.
                gf = gf.derive(param_var, 1)
                    * GenFun::var(param_var)
                    * GenFun::constant(lambda.clone() / T::from(k as u32));
            }
            let replacement = GenFun::constant((-lambda).exp()) * GenFun::var(param_var);
            let gf = gf.substitute_var(param_var, replacement);
            let result = gf.eval_with(inputs, degree_p1, cache);
            result.truncate_to_degree_p1(degree_p1)
        } else if let Some((param_var, lambda, inner)) =
            recognize_continuous_poisson_observation(g, v)
        {
            // Similarly for continuous parameters to the Poisson distribution:
            // Normally, we would compute the n-th derivative of G(y + λ(x-1)) and evaluate it at x=0.
            // Instead, we can compute D^n(G) where D(G)(y) := λ G'(y), and evaluate it at y = y - λ.
            // Finally, we have to divide by n! to get the Taylor coefficient,
            // which we integrate into the loop to improve numerical stability.
            let mut gf = inner.clone();
            for k in 1..=order {
                // The division by `k` is for the factorial.
                gf = gf.derive(param_var, 1) * GenFun::constant(lambda.clone() / T::from(k as u32));
            }
            let replacement = GenFun::var(param_var) - GenFun::constant(lambda);
            let gf = gf.substitute_var(param_var, replacement);
            let result = gf.eval_with(inputs, degree_p1, cache);
            result.truncate_to_degree_p1(degree_p1)
        } else if let Some((param_var, p, inner)) = recognize_negative_binomial_observation(g, v) {
            let mut lahs_cur = vec![T::one(); 1]; // Row of Lah numbers L_{d,i} for fixed d, multiplied by (1-p)^d / d!
            let one_mp = T::one() - p.clone();
            // Compute the d-th row of Lah numbers multiplied by (1-p)^d / d!:
            for d in 1..=order {
                let mut lahs_next = Vec::new();
                for i in 0..=d {
                    let lah_dm1_i = lahs_cur.get(i).cloned().unwrap_or_else(T::zero);
                    let lah_dm1_im1 = if 1 <= i && i <= lahs_cur.len() {
                        lahs_cur[i - 1].clone()
                    } else {
                        T::zero()
                    };
                    let lah_d_i = one_mp.clone() / T::from(d as u32)
                        * (lah_dm1_i * T::from((d + i - 1) as u32) + lah_dm1_im1);
                    lahs_next.push(lah_d_i);
                }
                lahs_cur = lahs_next;
            }
            // Compute the Taylor polynomial, which consists of d+1 summands:
            let mut sum = TaylorPoly::zero_with(vec![degree_p1; inputs.len()]);
            let mut new_inputs = inputs.to_vec();
            new_inputs[param_var.id()] = p.clone() * inputs[param_var.id()].clone();
            let mut inner_result = inner.eval_with(&new_inputs, degree_p1 + order, cache);
            let mut p_param_var_power = TaylorPoly::one();
            let param_var_tp =
                TaylorPoly::var(param_var, inputs[param_var.id()].clone(), degree_p1);
            let p_param_var = TaylorPoly::from(p.clone()) * param_var_tp;
            for lah in &lahs_cur {
                // Add the i-th summand:
                let subst =
                    TaylorPoly::from(p.clone()) * TaylorPoly::var_at_zero(param_var, degree_p1);
                sum += inner_result.subst_var(param_var, &subst) // G^(i)(p*x)
                * p_param_var_power.clone() // (px)^i
                * TaylorPoly::from(lah.clone()); // L_{d,i} * (1-p)^d / d!
                p_param_var_power *= p_param_var.clone();
                inner_result = inner_result.derivative(param_var, 1);
            }
            // Result is \sum_{i=0}^d G^(i)(p*x) (px)^i L_{d,i} (1-p)^d / d!:
            sum.truncate_to_degree_p1(degree_p1)
        } else {
            let mut inputs = inputs.to_owned();
            let result = if v.id() == inputs.len() {
                inputs.push(T::zero());
                let taylor = g.eval_with(&inputs, degree_p1 + order, cache);
                taylor.coefficients_of_term(v, order).remove_last_variable()
            } else {
                inputs[v.id()] = T::zero();
                let taylor = g.eval_with(&inputs, degree_p1 + order, cache);
                taylor.coefficients_of_term(v, order)
            };
            result.truncate_to_degree_p1(degree_p1)
        }
    }

    pub fn to_computation(&self) -> SymGenFun<T> {
        match self {
            Self::Var(v) => SymGenFun::var(*v),
            Self::Const(x) => SymGenFun::lit(x.clone()),
            Self::Add(g, h) => g.to_computation() + h.to_computation(),
            Self::Neg(g) => -g.to_computation(),
            Self::Mul(g, h) => g.to_computation() * h.to_computation(),
            Self::Div(g, h) => g.to_computation() / h.to_computation(),
            Self::Polynomial(coeffs) => fold_coeffs(
                &mut SymGenFun::zero,
                &mut |x: &T| SymGenFun::lit(x.clone()),
                &mut |v, acc| acc * SymGenFun::var(v),
                &mut |a, b| a + b,
                &coeffs.view(),
            ),
            Self::Exp(g) => g.to_computation().exp(),
            Self::Log(g) => g.to_computation().log(),
            Self::Pow(g, exp) => g.to_computation().pow(*exp),
            Self::UniformMgf(g) => {
                let g_comp = g.to_computation();
                // TODO: this will yield to a division 0 / 0 if g_comp is 0.
                // Adjusting how division of Taylor expansions works might help.
                (g_comp.exp() - SymGenFun::one()) / g_comp
            }
            Self::Subst(g, v, subst) => {
                let evaluated_subst = subst.to_computation();
                g.to_computation().substitute_var(*v, evaluated_subst)
            }
            Self::Derivative(g, v, order) => {
                let v = *v;
                let mut deriv = g.to_computation();
                for _ in 0..*order {
                    deriv = deriv.derive(v);
                }
                deriv
            }
            Self::TaylorPolynomial(g, v, orders) => {
                let v = *v;
                let max_order = *orders.iter().max().unwrap_or(&0);
                let taylor = g.to_computation().taylor_coeffs(v, max_order);
                let mut keep_terms = vec![false; max_order + 1];
                for order in orders {
                    keep_terms[*order] = true;
                }
                (0..=max_order)
                    .rev()
                    .fold(SymGenFun::lit(T::zero()), |acc, i| {
                        if keep_terms[i] {
                            acc * SymGenFun::var(v) + taylor.coeff(i)
                        } else {
                            acc * SymGenFun::var(v)
                        }
                    })
            }
            Self::TaylorCoeffAtZero(g, v, order) => g
                .to_computation()
                .taylor_coeffs_at(*v, &T::zero(), *order)
                .coeff(*order),
            Self::TaylorCoeff(g, v, order) => {
                g.to_computation().taylor_coeffs(*v, *order).coeff(*order)
            }
        }
    }
}

impl<T: std::fmt::Display + Zero> std::fmt::Display for GeneratingFunctionKind<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_prec(0, f)
    }
}

fn recognize_discrete_poisson_observation<T: Number>(
    g: &GenFun<T>,
    aux_var: Var,
) -> Option<(Var, T, &GenFun<T>)> {
    use GeneratingFunctionKind::*;
    if let Subst(inner, param_var, g) = g.0.as_ref() {
        let param_var = *param_var;
        if let Mul(g, h) = g.0.as_ref() {
            if g != &GenFun::var(param_var) {
                return None;
            }
            if let Exp(h) = h.0.as_ref() {
                if let Mul(g, h) = h.0.as_ref() {
                    if let Const(lambda) = g.0.as_ref() {
                        if *h == GenFun::var(aux_var) - GenFun::constant(T::one()) {
                            return Some((param_var, lambda.clone(), inner));
                        }
                    }
                }
            }
        }
    }
    None
}

fn recognize_continuous_poisson_observation<T: Number>(
    g: &GenFun<T>,
    aux_var: Var,
) -> Option<(Var, T, &GenFun<T>)> {
    use GeneratingFunctionKind::*;
    if let Subst(inner, param_var, g) = g.0.as_ref() {
        let param_var = *param_var;
        if let Add(g, h) = g.0.as_ref() {
            if g != &GenFun::var(param_var) {
                return None;
            }
            if let Mul(g, h) = h.0.as_ref() {
                if let Const(lambda) = g.0.as_ref() {
                    if *h == GenFun::var(aux_var) - GenFun::constant(T::one()) {
                        return Some((param_var, lambda.clone(), inner));
                    }
                }
            }
        }
    }
    None
}

fn recognize_negative_binomial_observation<T: Number>(
    g: &GenFun<T>,
    aux_var: Var,
) -> Option<(Var, T, &GenFun<T>)> {
    use GeneratingFunctionKind::*;
    if let Subst(inner, param_var, g) = g.0.as_ref() {
        let param_var = *param_var;
        if let Mul(g, h) = g.0.as_ref() {
            if g != &GenFun::var(param_var) {
                return None;
            }
            if let Div(g, h) = h.0.as_ref() {
                let p = if let Const(p) = g.0.as_ref() {
                    p.clone()
                } else {
                    return None;
                };
                let expected =
                    GenFun::one() - GenFun::constant(T::one() - p.clone()) * GenFun::var(aux_var);
                if h == &expected {
                    return Some((param_var, p, inner));
                }
            }
        }
    }
    None
}

fn fold_coeffs<T, R>(
    zero: &mut impl FnMut() -> R,
    constant: &mut impl FnMut(&T) -> R,
    mul_var: &mut impl FnMut(Var, R) -> R,
    add: &mut impl FnMut(R, R) -> R,
    coeffs: &ndarray::ArrayViewD<T>,
) -> R {
    if coeffs.ndim() == 0 {
        return constant(coeffs.first().unwrap());
    }
    let v = coeffs.ndim() - 1;
    let mut result = zero();
    for coeff in coeffs.axis_iter(Axis(v)).rev() {
        result = mul_var(Var(v), result);
        let coeff = fold_coeffs(zero, constant, mul_var, add, &coeff);
        result = add(result, coeff);
    }
    result
}

#[must_use]
pub fn probs_taylor<T: Number>(
    pgf: &GenFun<T>,
    v: Var,
    var_info: &[SupportSet],
    max_n: usize,
) -> Vec<T> {
    assert!(
        var_info[v.id()].is_discrete(),
        "Can only compute probabilities for discrete variables"
    );
    let num_vars = var_info.len();
    let mut substs = (0..num_vars)
        .map(|i| {
            if var_info[i].is_discrete() {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect::<Vec<_>>();
    substs[v.id()] = T::zero();
    let expansion = pgf.eval(&substs, max_n + 1);
    let mut index = vec![0; num_vars];
    let mut probs = vec![];
    for i in 0..max_n {
        index[v.id()] = i;
        let prob = expansion.coefficient(&index);
        probs.push(prob);
    }
    probs
}

#[must_use]
pub fn moments_taylor<T: Number>(
    pgf: &GenFun<T>,
    v: Var,
    var_info: &[SupportSet],
    limit: usize,
) -> (T, Vec<T>) {
    let num_vars = var_info.len();
    let substs = (0..num_vars)
        .map(|i| {
            if var_info[i].is_discrete() {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect::<Vec<_>>();
    let expansion = pgf.eval(&substs, limit);
    let mut result = Vec::with_capacity(limit);
    let mut index = vec![0; num_vars];
    let mut factor = T::one();
    for i in 0..limit {
        index[v.id()] = i;
        result.push(expansion.coefficient(&index) * factor.clone());
        factor *= T::from((i + 1) as u32);
    }
    if var_info[v.id()].is_discrete() {
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

/// Given factorial moments (of order 0..), returns the total probability and the (raw) moments (of order 1..).
pub fn factorial_moments_to_moments<T: Number>(factorial_moments: &[T]) -> (T, Vec<T>) {
    // Use the Stirling numbers of second kind to compute the moments from the factorial moments.
    // See https://en.wikipedia.org/wiki/Factorial_moment#Calculation_of_moments
    let len = factorial_moments.len();
    let mut stirling_numbers = vec![vec![T::zero(); len]; len];
    for n in 0..len {
        stirling_numbers[n][0] = T::zero();
        stirling_numbers[n][n] = T::one();
        for k in 1..n {
            stirling_numbers[n][k] = stirling_numbers[n - 1][k - 1].clone()
                + T::from(k as u32) * stirling_numbers[n - 1][k].clone();
        }
    }
    let total = factorial_moments[0].clone();
    let mut moments = vec![T::zero(); len - 1];
    for n in 1..len {
        #[allow(clippy::needless_range_loop)]
        for k in 0..=n {
            moments[n - 1] += stirling_numbers[n][k].clone() * factorial_moments[k].clone();
        }
    }
    for moment in &mut moments {
        *moment /= total.clone();
    }
    (total, moments)
}

/// Given the raw moments, returns the mean and the central moments of orders 2, 3, 4.
pub fn moments_to_central_moments<T: Number>(moments: &[T]) -> (T, Vec<T>) {
    let len = moments.len() + 1;
    let mean = moments[0].clone();
    let mut binomial_coeffs = vec![vec![T::zero(); len]; len];
    for n in 0..len {
        binomial_coeffs[n][0] = T::one();
        binomial_coeffs[n][n] = T::one();
        for k in 1..n {
            binomial_coeffs[n][k] =
                binomial_coeffs[n - 1][k - 1].clone() + binomial_coeffs[n - 1][k].clone();
        }
    }
    let neg_mean = -mean.clone();
    let mut central_moments = vec![T::zero(); len - 2];
    for n in 2..len {
        for k in 1..=n {
            central_moments[n - 2] += binomial_coeffs[n][k].clone()
                * (neg_mean).pow(n as u32 - k as u32)
                * moments[k - 1].clone();
        }
        central_moments[n - 2] += neg_mean.pow(n as u32);
    }
    (mean, central_moments)
}

/// Given the central moments (of order 2..), returns the variance and standardized moments (of order 3..)
///
/// Returns `(variance, [skewness, kurtosis])`.
pub fn central_to_standardized_moments<T: FloatNumber>(central_moments: &[T]) -> (T, Vec<T>) {
    let variance = central_moments[0].clone();
    let sigma = variance.sqrt();
    let result = central_moments
        .iter()
        .skip(1)
        .enumerate()
        .map(|(i, x)| {
            if x.is_zero() && !variance.is_nan() && !variance.is_zero() {
                x.clone()
            } else {
                let sigma_power = if i % 2 == 0 {
                    sigma.pow((i + 3) as u32)
                } else {
                    // Special case to avoid square roots (useful for rational computations)
                    variance.pow(((i + 3) / 2) as u32)
                };
                x.clone() / sigma_power
            }
        })
        .collect();
    (variance, result)
}
