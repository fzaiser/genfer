use std::borrow::Borrow;
use std::fmt::Display;

use crate::{numbers::Number, support::SupportSet};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Natural(pub u64);

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

    pub fn as_integer(&self) -> Option<u64> {
        if self.denom != 0 && self.numer % self.denom == 0 {
            Some(self.numer / self.denom)
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
pub enum Distribution {
    Dirac(Natural),
    Bernoulli(PosRatio),
    Binomial(Natural, PosRatio),
    Categorical(Vec<PosRatio>),
    NegBinomial(Natural, PosRatio),
    Geometric(PosRatio),
    /// Uniform distribution on the integers {start, ..., end - 1}
    Uniform {
        start: Natural,
        end: Natural,
    },
}

impl Distribution {
    pub fn support(&self) -> SupportSet {
        match self {
            Distribution::Dirac(a) => SupportSet::point(a.0),
            Distribution::Bernoulli(_) => (0..=1).into(),
            Distribution::Binomial(n, _) => (0..=n.0).into(),
            Distribution::Categorical(rs) => (0..rs.len() as u64).into(),
            Distribution::NegBinomial(..) | Distribution::Geometric(_) => SupportSet::naturals(),
            Distribution::Uniform { start, end } => (start.0..end.0).into(),
        }
    }
}

impl Display for Distribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Distribution::Dirac(a) => write!(f, "Dirac({a})"),
            Distribution::Bernoulli(p) => write!(f, "Bernoulli({p})"),
            Distribution::Binomial(n, p) => write!(f, "Binomial({n}, {p})"),
            Distribution::Categorical(rs) => {
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
            Distribution::NegBinomial(r, p) => write!(f, "NegBinomial({r}, {p})"),
            Distribution::Geometric(p) => write!(f, "Geometric({p})"),
            Distribution::Uniform { start, end } => write!(f, "Uniform({start}, {end})"),
        }
    }
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
            Event::DataFromDist(..) => VarRange::empty(),
            Event::Complement(e) => e.used_vars(),
            Event::Intersection(es) => es
                .iter()
                .fold(VarRange::empty(), |acc, e| acc.union(&e.used_vars())),
        }
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
    While {
        cond: Event,
        unroll: Option<usize>,
        body: Vec<Statement>,
    },
    Fail,
}

impl Statement {
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
            Statement::Sample {
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
            Statement::Assign {
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
                    if offset == &Natural(0) {
                        writeln!(f, ";")?;
                    } else {
                        writeln!(f, " + {offset};")?;
                    }
                } else {
                    writeln!(f, "{offset};")?;
                }
                Ok(())
            }
            Statement::Decrement { var, offset } => writeln!(f, "{var} -= {offset};"),
            Statement::IfThenElse { cond, then, els } => {
                if let Some(event) = self.recognize_observe() {
                    return writeln!(f, "observe {event};");
                }
                writeln!(f, "if {cond} {{")?;
                Self::fmt_block(then, indent + 2, f)?;
                let indent_str = " ".repeat(indent);
                match els.as_slice() {
                    [] => writeln!(f, "{indent_str}}}")?,
                    [if_stmt @ Statement::IfThenElse { .. }] => {
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
            Statement::While { cond, unroll, body } => {
                let indent_str = " ".repeat(indent);
                write!(f, "while {cond} ")?;
                if let Some(unroll) = unroll {
                    write!(f, "unroll {unroll} ")?;
                }
                writeln!(f, "{{")?;
                Self::fmt_block(body, indent + 2, f)?;
                writeln!(f, "{indent_str}}}")
            }
            Statement::Fail => writeln!(f, "fail;"),
        }
    }

    pub fn uses_observe(&self) -> bool {
        match self {
            Statement::Sample { .. } | Statement::Assign { .. } | Statement::Decrement { .. } => {
                false
            }
            Statement::IfThenElse { then, els, .. } => {
                then.iter().any(Statement::uses_observe) || els.iter().any(Statement::uses_observe)
            }
            Statement::While { body, .. } => body.iter().any(Statement::uses_observe),
            Statement::Fail => true,
        }
    }

    pub fn used_vars(&self) -> VarRange {
        match self {
            Statement::Sample { var: v, .. } | Statement::Decrement { var: v, offset: _ } => {
                VarRange::new(*v)
            }
            Statement::Assign {
                var: v, addend: a, ..
            } => VarRange::new(*v).union(&if let Some((_, w)) = a {
                VarRange::new(*w)
            } else {
                VarRange::empty()
            }),
            Statement::IfThenElse { cond, then, els } => cond
                .used_vars()
                .union(&VarRange::union_all(then.iter().map(Statement::used_vars)))
                .union(&VarRange::union_all(els.iter().map(Statement::used_vars))),
            Statement::While {
                cond,
                body,
                unroll: _,
            } => cond
                .used_vars()
                .union(&VarRange::union_all(body.iter().map(Statement::used_vars))),
            Statement::Fail => VarRange::empty(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Statement::Sample { .. }
            | Statement::Assign { .. }
            | Statement::Decrement { .. }
            | Statement::Fail => 1,
            Statement::IfThenElse { then, els, .. } => {
                1 + then.iter().fold(0, |acc, stmt| acc + stmt.size())
                    + els.iter().fold(0, |acc, stmt| acc + stmt.size())
            }
            Statement::While { body, .. } => 1 + body.iter().fold(0, |acc, stmt| acc + stmt.size()),
        }
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.indent_fmt(0, f)
    }
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

    pub fn size(&self) -> usize {
        self.stmts.iter().fold(0, |acc, stmt| acc + stmt.size())
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Statement::fmt_block(&self.stmts, 0, f)?;
        write!(f, "return {}", self.result)
    }
}
