use std::borrow::Borrow;
use std::fmt::Display;

use crate::{
    generating_function::GenFun,
    number::{Number, Rational},
    support::SupportSet,
};
use Distribution::*;
use Statement::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Natural(pub u32);

impl Display for Natural {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::ops::AddAssign for Natural {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::Add for Natural {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PosRatio {
    pub numer: u64,
    pub denom: u64,
}

impl PosRatio {
    pub fn new(numer: u64, denom: u64) -> Self {
        Self { numer, denom }
    }

    pub fn zero() -> Self {
        Self::new(0, 1)
    }

    pub fn is_zero(&self) -> bool {
        self.numer == 0 && self.denom != 0
    }

    pub fn infinity() -> Self {
        Self::new(1, 0)
    }

    pub fn complement(&self) -> Self {
        assert!(self.numer <= self.denom);
        Self::new(self.denom - self.numer, self.denom)
    }

    pub fn as_integer(&self) -> Option<u32> {
        if self.denom != 0 && self.numer % self.denom == 0 {
            u32::try_from(self.numer / self.denom).ok()
        } else {
            None
        }
    }

    pub fn round(&self) -> f64 {
        (self.numer as f64) / (self.denom as f64)
    }
}

impl From<u64> for PosRatio {
    fn from(n: u64) -> Self {
        Self::new(n, 1)
    }
}

impl From<u32> for PosRatio {
    fn from(n: u32) -> Self {
        Self::from(u64::from(n))
    }
}

impl Display for PosRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denom == 1 {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "{}/{}", self.numer, self.denom)
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Var(pub usize);

impl Var {
    #[inline]
    pub fn id(&self) -> usize {
        self.0
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let i = self.0;
        if i < 26 {
            let var = ('a' as usize + i) as u8 as char;
            write!(f, "{var}")
        } else {
            write!(f, "x_{i}")
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarRange(usize);

impl VarRange {
    #[inline]
    pub fn new(var: Var) -> Self {
        Self(var.id() + 1)
    }

    #[inline]
    pub fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub fn add(&self, var: Var) -> Self {
        self.union(&VarRange::new(var))
    }

    #[inline]
    pub fn union(&self, other: &VarRange) -> Self {
        Self(self.0.max(other.0))
    }

    #[inline]
    pub fn remove(&self, var: Var) -> Self {
        if var.id() + 1 == self.0 {
            Self(var.id())
        } else {
            self.clone()
        }
    }

    #[inline]
    pub fn max_var(&self) -> Option<Var> {
        self.0.checked_sub(1).map(Var)
    }

    #[inline]
    pub fn union_all(iter: impl Iterator<Item = impl Borrow<VarRange>>) -> Self {
        let mut result = VarRange::empty();
        for varset in iter {
            result = result.union(varset.borrow());
        }
        result
    }

    #[inline]
    pub fn num_vars(&self) -> usize {
        self.0
    }

    #[inline]
    pub fn zero_to(num_vars: usize) -> VarRange {
        VarRange(num_vars)
    }
}

#[derive(Clone, Debug)]
pub struct GfTranslation<T> {
    pub var_info: Vec<SupportSet>,
    pub gf: GenFun<T>,
}

#[derive(Clone, Debug)]
pub enum Distribution {
    Dirac(PosRatio),
    Bernoulli(PosRatio),
    BernoulliVarProb(Var),
    BinomialVarTrials(Var, PosRatio),
    Binomial(Natural, PosRatio),
    Categorical(Vec<PosRatio>),
    NegBinomialVarSuccesses(Var, PosRatio),
    NegBinomial(Natural, PosRatio),
    Geometric(PosRatio),
    Poisson(PosRatio),
    PoissonVarRate(PosRatio, Var),
    /// Uniform distribution on the integers {start, ..., end - 1}
    Uniform {
        start: Natural,
        end: Natural,
    },
    Exponential {
        rate: PosRatio,
    },
    Gamma {
        shape: PosRatio,
        rate: PosRatio,
    },
    UniformCont {
        start: PosRatio,
        end: PosRatio,
    },
}

impl Distribution {
    fn support(&self) -> SupportSet {
        use Distribution::*;
        match self {
            Dirac(a) => {
                if let Some(a) = a.as_integer() {
                    SupportSet::point(a)
                } else {
                    SupportSet::interval(
                        Rational::from_ratio(a.numer, a.denom),
                        Rational::from_ratio(a.numer, a.denom),
                    )
                }
            }
            Bernoulli(_) | BernoulliVarProb(_) => (0..=1).into(),
            Binomial(n, _) => (0..=n.0).into(),
            Categorical(rs) => (0..rs.len() as u32).into(),
            BinomialVarTrials(..)
            | NegBinomialVarSuccesses(..)
            | NegBinomial(..)
            | Geometric(_)
            | Poisson(_)
            | PoissonVarRate(..) => SupportSet::naturals(),
            Uniform { start, end } => (start.0..end.0).into(),
            Exponential { .. } | Gamma { .. } => SupportSet::nonneg_reals(),
            UniformCont { start, end } => SupportSet::interval(
                Rational::from_ratio(start.numer, start.denom),
                Rational::from_ratio(end.numer, end.denom),
            ),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn transform_gf<T: Number>(
        &self,
        v: Var,
        translation: GfTranslation<T>,
        add_previous_value: bool,
    ) -> GfTranslation<T> {
        use Distribution::*;
        let base = if add_previous_value {
            translation.gf.clone()
        } else {
            marginalize_out(v, &translation.gf, &translation.var_info)
        };
        let mut new_var_info = translation.var_info.clone();
        if v.id() == new_var_info.len() {
            new_var_info.push(SupportSet::zero());
        }
        assert!(v.id() < new_var_info.len());
        if !add_previous_value {
            new_var_info[v.id()] = SupportSet::zero();
        }
        new_var_info[v.id()] += self.support();
        let gf = &translation.gf;
        let gf = match self {
            Dirac(a) => {
                let dirac = if let Some(a) = a.as_integer() {
                    GenFun::var(v).pow(a)
                } else {
                    (GenFun::var(v) * GenFun::from_ratio(*a)).exp()
                };
                dirac * base
            }
            Bernoulli(p) => {
                let bernoulli =
                    GenFun::from_ratio(*p) * GenFun::var(v) + GenFun::from_ratio(p.complement());
                bernoulli * base
            }
            BernoulliVarProb(w) => {
                let prob_times_gf = if translation.var_info[w.id()].is_discrete() {
                    gf.derive(*w, 1) * GenFun::var(*w)
                } else {
                    gf.derive(*w, 1)
                };
                let prob_times_base = if add_previous_value {
                    prob_times_gf
                } else {
                    marginalize_out(v, &prob_times_gf, &translation.var_info)
                };
                let v_term = if new_var_info[v.id()].is_discrete() {
                    GenFun::var(v)
                } else {
                    GenFun::var(v).exp()
                };
                base + (v_term - GenFun::one()) * prob_times_base
            }
            BinomialVarTrials(w, p) => {
                let subst =
                    GenFun::from_ratio(*p) * GenFun::var(v) + GenFun::from_ratio(p.complement());
                Self::compound_dist(gf, &base, v, *w, add_previous_value, true, subst)
            }
            Binomial(n, p) => {
                let binomial = (GenFun::from_ratio(*p) * GenFun::var(v)
                    + GenFun::from_ratio(p.complement()))
                .pow(n.0);
                binomial * base
            }
            Categorical(rs) => {
                let mut categorical = GenFun::zero();
                for r in rs.iter().rev() {
                    categorical *= GenFun::var(v);
                    categorical += GenFun::from_ratio(*r);
                }
                categorical * base
            }
            NegBinomialVarSuccesses(w, p) => {
                let subst = GenFun::from_ratio(*p)
                    / (GenFun::one() - GenFun::from_ratio(p.complement()) * GenFun::var(v));
                Self::compound_dist(gf, &base, v, *w, add_previous_value, true, subst)
            }
            NegBinomial(n, p) => {
                let geometric = GenFun::from_ratio(*p)
                    / (GenFun::one() - GenFun::from_ratio(p.complement()) * GenFun::var(v));
                geometric.pow(n.0) * base
            }
            Geometric(p) => {
                let geometric = GenFun::from_ratio(*p)
                    / (GenFun::one() - GenFun::from_ratio(p.complement()) * GenFun::var(v));
                geometric * base
            }
            Poisson(lambda) => {
                let poisson =
                    (GenFun::from_ratio(*lambda) * (GenFun::var(v) - GenFun::one())).exp();
                poisson * base
            }
            PoissonVarRate(lambda, w) => {
                let w_discrete = translation.var_info[w.id()].is_discrete();
                let subst = if w_discrete {
                    (GenFun::from_ratio(*lambda) * (GenFun::var(v) - GenFun::one())).exp()
                } else {
                    GenFun::from_ratio(*lambda) * (GenFun::var(v) - GenFun::one())
                };
                Self::compound_dist(gf, &base, v, *w, add_previous_value, w_discrete, subst)
            }
            Uniform { start, end } => {
                let mut uniform = GenFun::zero();
                assert!(end.0 > start.0, "Uniform distribution cannot have length 0");
                let length = end.0 - start.0;
                let weight = GenFun::from_ratio(PosRatio::new(1, u64::from(length)));
                for _ in 0..length {
                    uniform = weight.clone() + GenFun::var(v) * uniform;
                }
                uniform *= GenFun::var(v).pow(start.0);
                uniform * base
            }
            Exponential { rate } => {
                let beta = GenFun::from_ratio(*rate);
                let exponential = beta.clone() / (beta - GenFun::var(v));
                exponential * base
            }
            Gamma { shape, rate } => {
                let beta = GenFun::from_ratio(*rate);
                let gamma = if let Some(shape) = shape.as_integer() {
                    // Optimized representation avoiding logarithms for integer exponents
                    (beta.clone() / (beta - GenFun::var(v))).pow(shape)
                } else {
                    (GenFun::from_ratio(*shape) * (beta.log() - (beta - GenFun::var(v)).log()))
                        .exp()
                };
                gamma * base
            }
            UniformCont { start, end } => {
                let width =
                    T::from_ratio(end.numer, end.denom) - T::from_ratio(start.numer, start.denom);
                let x = GenFun::constant(width) * GenFun::var(v);
                let uniform =
                    GenFun::uniform_mgf(x) * (GenFun::from_ratio(*start) * GenFun::var(v)).exp();
                uniform * base
            }
        };
        GfTranslation {
            gf,
            var_info: new_var_info,
        }
    }

    fn compound_dist<T: Number>(
        gf: &GenFun<T>,
        base: &GenFun<T>,
        sampled_var: Var,
        param_var: Var,
        add_previous_value: bool,
        param_var_discrete: bool,
        subst: GenFun<T>,
    ) -> GenFun<T> {
        if sampled_var == param_var {
            if add_previous_value {
                let substitution = if param_var_discrete {
                    GenFun::var(param_var) * subst
                } else {
                    GenFun::var(param_var) + subst
                };
                gf.substitute_var(param_var, substitution)
            } else {
                gf.substitute_var(param_var, subst)
            }
        } else {
            let substitution = if param_var_discrete {
                GenFun::var(param_var) * subst
            } else {
                GenFun::var(param_var) + subst
            };
            base.substitute_var(param_var, substitution)
        }
    }

    fn used_vars(&self) -> VarRange {
        match self {
            Dirac(_)
            | Bernoulli(_)
            | Binomial(_, _)
            | Categorical(_)
            | NegBinomial(_, _)
            | Geometric(_)
            | Poisson(_)
            | Uniform { .. }
            | Exponential { .. }
            | Gamma { .. }
            | UniformCont { .. } => VarRange::empty(),
            BernoulliVarProb(v)
            | BinomialVarTrials(v, _)
            | NegBinomialVarSuccesses(v, _)
            | PoissonVarRate(_, v) => VarRange::new(*v),
        }
    }
}

impl Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dirac(a) => write!(f, "Dirac({a})"),
            Bernoulli(p) => write!(f, "Bernoulli({p})"),
            BernoulliVarProb(v) => write!(f, "Bernoulli({v})"),
            BinomialVarTrials(n, p) => write!(f, "Binomial({n}, {p})"),
            Binomial(n, p) => write!(f, "Binomial({n}, {p})"),
            Categorical(rs) => {
                write!(f, "Categorical(")?;
                let mut first = true;
                for r in rs {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{r}")?;
                }
                write!(f, ")")
            }
            NegBinomialVarSuccesses(r, p) => write!(f, "NegBinomial({r}, {p})"),
            NegBinomial(r, p) => write!(f, "NegBinomial({r}, {p})"),
            Geometric(p) => write!(f, "Geometric({p})"),
            Poisson(lambda) => write!(f, "Poisson({lambda})"),
            PoissonVarRate(lambda, n) => write!(f, "Poisson({lambda} * {n})"),
            Uniform { start, end } => write!(f, "Uniform({start}, {end})"),
            Exponential { rate } => write!(f, "Exponential({rate})"),
            Gamma { shape, rate } => write!(f, "Gamma({shape}, {rate})"),
            UniformCont { start, end } => write!(f, "UniformCont({start}, {end})"),
        }
    }
}

fn marginalize_out<T: Number>(v: Var, gf: &GenFun<T>, var_info: &[SupportSet]) -> GenFun<T> {
    if v.id() >= var_info.len() {
        // This can only be a temporary variable of index n, where n is the number of variables.
        // This is introduced for observe c ~ D(X_i) statements for the temporary variable X_n ~ D(X_i).
        assert!(v.id() == var_info.len());
        // In this case, the variable is discrete because we only support discrete observations.
        gf.substitute_var(v, GenFun::one())
    } else if var_info[v.id()].is_discrete() {
        gf.substitute_var(v, GenFun::one())
    } else {
        gf.substitute_var(v, GenFun::zero())
    }
}

fn marginalize_all<T: Number>(gf: GenFun<T>, var_info: &[SupportSet]) -> GenFun<T> {
    let mut result = gf;
    for v in 0..var_info.len() {
        result = marginalize_out(Var(v), &result, var_info);
    }
    result
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Comparison {
    Eq,
    Lt,
    Le,
}

impl Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Comparison::Eq => write!(f, "="),
            Comparison::Lt => write!(f, "<"),
            Comparison::Le => write!(f, "<="),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    InSet(Var, Vec<Natural>),
    VarComparison(Var, Comparison, Var),
    DataFromDist(Natural, Distribution),
    Complement(Box<Event>),
    Intersection(Vec<Event>),
}

impl Event {
    pub fn used_vars(&self) -> VarRange {
        match self {
            Event::InSet(v, _) => VarRange::new(*v),
            Event::VarComparison(v1, _, v2) => VarRange::new(*v1).add(*v2),
            Event::DataFromDist(_, dist) => dist.used_vars(),
            Event::Complement(e) => e.used_vars(),
            Event::Intersection(es) => es
                .iter()
                .fold(VarRange::empty(), |acc, e| acc.union(&e.used_vars())),
        }
    }

    pub fn transform_gf<T>(&self, translation: GfTranslation<T>) -> GfTranslation<T>
    where
        T: Number,
    {
        fn gf_in_set<T: Number>(var: Var, set: &[Natural], gf: GenFun<T>) -> GenFun<T> {
            if let &[order] = &set {
                let order = order.0;
                gf.taylor_coeff_at_zero(var, order as usize) * GenFun::var(var).pow(order)
            } else {
                gf.taylor_polynomial_at_zero(var, set.iter().map(|n| n.0 as usize).collect())
            }
        }
        let var_info = translation.var_info.clone();
        let gf = translation.gf.clone();
        let gf = match self {
            Event::InSet(var, set) => gf_in_set(*var, set, gf),
            Event::VarComparison(v1, comp, v2) => {
                let (scrutinee, other, reversed, range) = match (
                    var_info[v1.id()].finite_nonempty_range(),
                    var_info[v2.id()].finite_nonempty_range(),
                ) {
                    (None, None) => panic!("Cannot compare two variables with infinite support."),
                    (None, Some(r)) => (*v2, *v1, false, r),
                    (Some(r), None) => (*v1, *v2, true, r),
                    (Some(r1), Some(r2)) => {
                        if r1.end() - r1.start() <= r2.end() - r2.start() {
                            (*v1, *v2, true, r1)
                        } else {
                            (*v2, *v1, false, r2)
                        }
                    }
                };
                let mut result = GenFun::zero();
                for i in range {
                    let gf_scrutinee_eq_i = gf_in_set(scrutinee, &[Natural(i)], gf.clone());
                    let summand = match (comp, reversed) {
                        (Comparison::Eq, _) => gf_in_set(other, &[Natural(i)], gf_scrutinee_eq_i),
                        (Comparison::Lt, false) => gf_in_set(
                            other,
                            &(0..i).map(Natural).collect::<Vec<_>>(),
                            gf_scrutinee_eq_i,
                        ),
                        (Comparison::Lt, true) => {
                            gf_scrutinee_eq_i.clone()
                                - gf_in_set(
                                    other,
                                    &(0..=i).map(Natural).collect::<Vec<_>>(),
                                    gf_scrutinee_eq_i,
                                )
                        }
                        (Comparison::Le, false) => gf_in_set(
                            other,
                            &(0..=i).map(Natural).collect::<Vec<_>>(),
                            gf_scrutinee_eq_i,
                        ),
                        (Comparison::Le, true) => {
                            gf_scrutinee_eq_i.clone()
                                - gf_in_set(
                                    other,
                                    &(0..i).map(Natural).collect::<Vec<_>>(),
                                    gf_scrutinee_eq_i,
                                )
                        }
                    };
                    result += summand;
                }
                result
            }
            Event::DataFromDist(data, dist) => {
                if let Some(factor) = self.recognize_const_prob() {
                    GenFun::constant(factor) * gf
                } else {
                    Event::transform_gf_data_from_dist(*data, dist, &var_info, gf)
                }
            }
            Event::Complement(e) => gf - e.transform_gf(translation).gf,
            Event::Intersection(es) => {
                let mut translation = translation;
                for e in es {
                    translation = e.transform_gf(translation);
                }
                translation.gf
            }
        };
        GfTranslation { var_info, gf }
    }

    pub fn recognize_const_prob<T: Number>(&self) -> Option<T> {
        match self {
            Event::InSet(..) | Event::VarComparison(..) => None,
            Event::DataFromDist(data, dist) => match dist {
                Distribution::Bernoulli(p) => match data.0 {
                    0 => {
                        let p1 = p.complement();
                        Some(T::from_ratio(p1.numer, p1.denom))
                    }
                    1 => Some(T::from_ratio(p.numer, p.denom)),
                    _ => Some(T::zero()),
                },
                _ => None,
            },
            Event::Complement(e) => e.recognize_const_prob().map(|p| T::one() - p),
            Event::Intersection(es) => {
                let mut result = T::one();
                for e in es {
                    result *= e.recognize_const_prob()?;
                }
                Some(result)
            }
        }
    }

    fn transform_gf_data_from_dist<T: Number>(
        data: Natural,
        dist: &Distribution,
        var_info: &[SupportSet],
        gf: GenFun<T>,
    ) -> GenFun<T> {
        match dist {
            Distribution::BernoulliVarProb(var) => {
                let prob_times_gf = if var_info[var.id()].is_discrete() {
                    gf.derive(*var, 1) * GenFun::var(*var)
                } else {
                    gf.derive(*var, 1)
                };
                match data.0 {
                    0 => gf - prob_times_gf,
                    1 => prob_times_gf,
                    _ => GenFun::zero(),
                }
            }
            Distribution::BinomialVarTrials(var, p) => {
                let order = data.0 as usize;
                let replacement = GenFun::from_ratio(p.complement()) * GenFun::var(*var);
                gf.taylor_coeff(*var, order)
                    .substitute_var(*var, replacement)
                    * (GenFun::from_ratio(*p) * GenFun::var(*var)).pow(data.0)
            }
            _ => {
                // TODO: this can be optimized for distributions that only have constant parameters.
                // In that case, we should just multiply by the probability mass function.
                let new_var = Var(gf.used_vars().num_vars());
                let new_translation = Statement::Sample {
                    var: new_var,
                    distribution: dist.clone(),
                    add_previous_value: false,
                }
                .transform_gf(GfTranslation {
                    var_info: var_info.to_vec(),
                    gf,
                });
                let gf = new_translation
                    .gf
                    .taylor_coeff_at_zero(new_var, data.0 as usize);
                marginalize_out(new_var, &gf, &new_translation.var_info)
            }
        }
    }

    pub fn complement(self) -> Event {
        if let Event::Complement(e) = self {
            *e
        } else {
            Event::Complement(Box::new(self))
        }
    }

    pub fn and(self, other: Event) -> Event {
        match (self, other) {
            (Event::Intersection(mut es), Event::Intersection(mut es2)) => {
                es.append(&mut es2);
                Event::Intersection(es)
            }
            (Event::Intersection(mut es), e) => {
                es.push(e);
                Event::Intersection(es)
            }
            (e, Event::Intersection(mut es)) => {
                es.insert(0, e);
                Event::Intersection(es)
            }
            (e1, e2) => Event::Intersection(vec![e1, e2]),
        }
    }

    pub fn intersection(es: Vec<Event>) -> Event {
        let mut conjuncts = Vec::new();
        for e in es {
            if let Event::Intersection(mut es) = e {
                conjuncts.append(&mut es);
            } else {
                conjuncts.push(e);
            }
        }
        if conjuncts.len() == 1 {
            conjuncts.pop().unwrap()
        } else {
            Event::Intersection(conjuncts)
        }
    }

    pub fn disjunction(es: Vec<Event>) -> Event {
        if es.len() == 1 {
            es[0].clone()
        } else {
            Event::intersection(es.into_iter().map(Event::complement).collect()).complement()
        }
    }

    pub fn always() -> Event {
        Event::intersection(Vec::new())
    }

    pub fn never() -> Event {
        Event::always().complement()
    }
}

impl Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Event::InSet(var, set) => write!(
                f,
                "{var} ∈ {:?}",
                set.iter().map(|n| n.0).collect::<Vec<_>>()
            ),
            Event::VarComparison(v1, comp, v2) => write!(f, "{v1} {comp} {v2}"),
            Event::DataFromDist(data, dist) => write!(f, "{data} ~ {dist}"),
            Event::Complement(e) => write!(f, "not ({e})"),
            Event::Intersection(es) => {
                let mut first = true;
                for e in es {
                    if !first {
                        write!(f, " and ")?;
                    }
                    first = false;
                    write!(f, "{e}")?;
                }
                if first {
                    write!(f, "true")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Statement {
    /// Sample a variable from a distribution and add the previous value of the variable if true.
    Sample {
        var: Var,
        distribution: Distribution,
        add_previous_value: bool,
    },
    Assign {
        var: Var,
        add_previous_value: bool,
        addend: Option<(Natural, Var)>,
        offset: Natural,
    },
    Decrement {
        var: Var,
        offset: Natural,
    },
    IfThenElse {
        cond: Event,
        then: Vec<Statement>,
        els: Vec<Statement>,
    },
    Fail,
    Normalize {
        /// Variables that are fixed (not marginalized) for the normalization
        ///
        /// This is necessary for nested inference.
        given_vars: Vec<Var>,
        stmts: Vec<Statement>,
    },
}

impl Statement {
    pub fn transform_gf<T>(&self, translation: GfTranslation<T>) -> GfTranslation<T>
    where
        T: Number,
    {
        use Statement::*;
        match self {
            Sample {
                var: v,
                distribution,
                add_previous_value,
            } => distribution.transform_gf(*v, translation, *add_previous_value),
            Assign {
                var: v,
                add_previous_value,
                addend,
                offset,
            } => {
                let v = *v;
                let mut gf = translation.gf;
                let mut var_info = translation.var_info;
                let var = GenFun::var(v);
                let mut v_exp = if *add_previous_value { 1 } else { 0 };
                let mut new_support = if *add_previous_value {
                    var_info[v.id()].clone()
                } else {
                    SupportSet::zero()
                };
                let w_subst = if let Some((factor, w)) = addend {
                    let other_support = var_info[w.id()].clone();
                    new_support += other_support * factor.0;
                    if v == *w {
                        v_exp += factor.0;
                        None
                    } else if var_info[w.id()].is_discrete() {
                        Some((*w, GenFun::var(*w) * var.pow(factor.0)))
                    } else {
                        assert!(
                            !var_info[v.id()].is_discrete() || !*add_previous_value,
                            "cannot add a continuous to a discrete variable"
                        );
                        Some((
                            *w,
                            GenFun::var(*w) + var.clone() * GenFun::from_u32(factor.0),
                        ))
                    }
                } else {
                    None
                };
                if var_info[v.id()].is_discrete() {
                    gf = gf.substitute_var(v, var.pow(v_exp));
                } else {
                    gf = gf.substitute_var(v, var.clone() * GenFun::from_u32(v_exp));
                }
                if let Some((w, w_subst)) = w_subst {
                    gf = gf.substitute_var(w, w_subst);
                }
                new_support += SupportSet::from(offset.0);
                var_info[v.id()] = new_support;
                let gf = if var_info[v.id()].is_discrete() {
                    gf * var.pow(offset.0)
                } else {
                    gf * (var * GenFun::from_u32(offset.0)).exp()
                };
                GfTranslation { var_info, gf }
            }
            Decrement { var, offset } => {
                let v = *var;
                let gf = translation.gf;
                let mut var_info = translation.var_info;
                assert!(
                    var_info[v.id()].is_discrete(),
                    "cannot decrement continuous variables"
                );
                let new_support = var_info[v.id()].saturating_sub(offset.0);
                var_info[v.id()] = new_support;
                let gf = gf.shift_down_taylor_at_zero(v, offset.0 as usize);
                GfTranslation { var_info, gf }
            }
            IfThenElse { cond, then, els } => {
                if let Some(factor) = cond.recognize_const_prob::<T>() {
                    // In this case we can avoid path explosion by multiplying with factor
                    // AFTER transforming the if- and else-branches:
                    let translation_then = Self::transform_gf_program(then, translation.clone());
                    let translation_else = Self::transform_gf_program(els, translation);
                    GfTranslation {
                        var_info: join_var_infos(
                            &translation_then.var_info,
                            &translation_else.var_info,
                        ),
                        gf: GenFun::constant(factor.clone()) * translation_then.gf
                            + GenFun::constant(T::one() - factor) * translation_else.gf,
                    }
                } else {
                    let var_info = translation.var_info.clone();
                    let gf = translation.gf.clone();
                    let then_before = cond.transform_gf(translation);
                    let else_before = GfTranslation {
                        var_info,
                        gf: gf - then_before.gf.clone(),
                    };
                    let then_after = Self::transform_gf_program(then, then_before);
                    let else_after = Self::transform_gf_program(els, else_before);
                    GfTranslation {
                        var_info: join_var_infos(&then_after.var_info, &else_after.var_info),
                        gf: then_after.gf + else_after.gf,
                    }
                }
            }
            Fail => GfTranslation {
                var_info: translation.var_info,
                gf: GenFun::zero(),
            },
            Normalize {
                given_vars,
                stmts: block,
            } => Statement::normalize_with_vars(given_vars, block, translation),
        }
    }

    fn normalize_with_vars<T: Number>(
        given_vars: &[Var],
        block: &[Statement],
        translation: GfTranslation<T>,
    ) -> GfTranslation<T> {
        if given_vars.is_empty() {
            let total_before = marginalize_all(translation.gf.clone(), &translation.var_info);
            let translation = Self::transform_gf_program(block, translation);
            let total_after = marginalize_all(translation.gf.clone(), &translation.var_info);
            GfTranslation {
                var_info: translation.var_info,
                gf: total_before / total_after * translation.gf,
            }
        } else {
            let v = given_vars[0];
            let rest = &given_vars[1..];
            let mut var_info = translation.var_info.clone();
            let support = var_info[v.id()].clone();
            let range = support.finite_nonempty_range().unwrap_or_else(|| panic!("Cannot normalize with respect to variable `{v}`, because its value could not be proven to be bounded."));
            let mut gf = GenFun::zero();
            for i in range {
                let summand =
                    translation.gf.taylor_coeff_at_zero(v, i as usize) * GenFun::var(v).pow(i);
                let summand = GfTranslation {
                    var_info: translation.var_info.clone(),
                    gf: summand,
                };
                let result = Self::normalize_with_vars(rest, block, summand);
                gf += result.gf;
                var_info = join_var_infos(&var_info, &result.var_info);
            }
            GfTranslation { var_info, gf }
        }
    }

    fn transform_gf_program<T>(program: &[Statement], start: GfTranslation<T>) -> GfTranslation<T>
    where
        T: Number,
    {
        let mut current = start;
        for stmt in program {
            current = stmt.transform_gf(current);
        }
        current
    }

    fn fmt_block(
        stmts: &[Statement],
        indent: usize,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        for stmt in stmts {
            let indent_str = " ".repeat(indent);
            f.write_str(&indent_str)?;
            stmt.indent_fmt(indent, f)?;
        }
        Ok(())
    }

    pub fn recognize_observe(&self) -> Option<&Event> {
        if let Statement::IfThenElse { cond, then, els } = self {
            if then.is_empty() && matches!(els.as_slice(), &[Statement::Fail]) {
                return Some(cond);
            }
        }
        None
    }

    fn indent_fmt(&self, indent: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Sample {
                var,
                distribution: dist,
                add_previous_value: add_previous,
            } => {
                if *add_previous {
                    writeln!(f, "{var} +~ {dist};")
                } else {
                    writeln!(f, "{var} ~ {dist};")
                }
            }
            Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                if *add_previous_value {
                    write!(f, "{var} += ")?;
                } else {
                    write!(f, "{var} := ")?;
                }
                if let Some((coeff, var)) = addend {
                    if *coeff != Natural(1) {
                        write!(f, "{coeff} * ")?;
                    }
                    write!(f, "{var}")?;
                    if offset != &Natural(0) {
                        writeln!(f, " + {offset};")?;
                    } else {
                        writeln!(f, ";")?;
                    }
                } else {
                    writeln!(f, "{offset};")?;
                }
                Ok(())
            }
            Decrement { var, offset } => writeln!(f, "{var} -= {offset};"),
            IfThenElse { cond, then, els } => {
                if let Some(event) = self.recognize_observe() {
                    return writeln!(f, "observe {event};");
                }
                writeln!(f, "if {cond} {{")?;
                Self::fmt_block(then, indent + 2, f)?;
                let indent_str = " ".repeat(indent);
                match els.as_slice() {
                    [] => writeln!(f, "{indent_str}}}")?,
                    [if_stmt @ IfThenElse { .. }] => {
                        write!(f, "{indent_str}}} else ")?;
                        if_stmt.indent_fmt(indent, f)?;
                    }
                    _ => {
                        writeln!(f, "{indent_str}}} else {{")?;
                        Self::fmt_block(els, indent + 2, f)?;
                        writeln!(f, "{indent_str}}}")?;
                    }
                }
                Ok(())
            }
            Fail => writeln!(f, "fail;"),
            Normalize { given_vars, stmts } => {
                let indent_str = " ".repeat(indent);
                write!(f, "normalize")?;
                for v in given_vars {
                    write!(f, " {v}")?;
                }
                writeln!(f, " {{")?;
                Self::fmt_block(stmts, indent + 2, f)?;
                writeln!(f, "{indent_str}}}")
            }
        }
    }

    pub fn uses_observe(&self) -> bool {
        match self {
            Sample { .. } | Assign { .. } | Decrement { .. } => false,
            IfThenElse { then, els, .. } => {
                then.iter().any(Statement::uses_observe) || els.iter().any(Statement::uses_observe)
            }
            Fail => true,
            Normalize { stmts, .. } => stmts.iter().any(Statement::uses_observe),
        }
    }

    pub fn used_vars(&self) -> VarRange {
        match self {
            Sample {
                var: v,
                distribution: d,
                add_previous_value: _,
            } => d.used_vars().add(*v),
            Assign {
                var: v, addend: a, ..
            } => VarRange::new(*v).union(&if let Some((_, w)) = a {
                VarRange::new(*w)
            } else {
                VarRange::empty()
            }),
            Decrement { var: v, offset: _ } => VarRange::new(*v),
            IfThenElse { cond, then, els } => cond
                .used_vars()
                .union(&VarRange::union_all(then.iter().map(Statement::used_vars)))
                .union(&VarRange::union_all(els.iter().map(Statement::used_vars))),
            Fail => VarRange::empty(),
            Normalize {
                given_vars: _,
                stmts,
            } => VarRange::union_all(stmts.iter().map(Statement::used_vars)),
        }
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.indent_fmt(0, f)
    }
}

fn join_var_infos(var_info_then: &[SupportSet], var_info_else: &[SupportSet]) -> Vec<SupportSet> {
    let mut var_info = Vec::new();
    let len = var_info_then.len().max(var_info_else.len());
    for i in 0..len {
        match (var_info_then.get(i), var_info_else.get(i)) {
            (Some(set), None) | (None, Some(set)) => var_info.push(set.clone()),
            (Some(set1), Some(set2)) => {
                var_info.push(set1.join(set2));
            }
            (None, None) => panic!("This should not happen"),
        }
    }
    var_info
}

#[derive(Clone, Debug)]
pub struct Program {
    pub stmts: Vec<Statement>,
    pub result: Var,
}

impl Program {
    pub const fn new(stmts: Vec<Statement>, result: Var) -> Self {
        Program { stmts, result }
    }

    pub fn uses_observe(&self) -> bool {
        self.stmts.iter().any(Statement::uses_observe)
    }

    pub fn used_vars(&self) -> VarRange {
        VarRange::union_all(self.stmts.iter().map(Statement::used_vars))
    }

    pub fn transform_gf<T>(&self, gf: GenFun<T>) -> GfTranslation<T>
    where
        T: Number,
    {
        let num_vars = self.used_vars().num_vars();
        let var_info = vec![SupportSet::zero(); num_vars];
        Statement::transform_gf_program(&self.stmts, GfTranslation { var_info, gf })
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Statement::fmt_block(&self.stmts, 0, f)?;
        write!(f, "return {}", self.result)
    }
}
