use num_traits::{One, Zero};

use crate::{numbers::Rational, ppl::Var, semantics::support::VarSupport, support::SupportSet};

use super::FiniteDiscrete;

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBound {
    /// Lower bound on the probability mass function
    pub lower: FiniteDiscrete,
    /// Lower bound on the rejection probability
    pub reject: Rational,
    /// Overapproximation of the support set of the program distribution
    pub var_supports: VarSupport,
}

impl ResidualBound {
    pub(crate) fn zero(n: usize) -> ResidualBound {
        ResidualBound {
            lower: FiniteDiscrete::zero(n),
            reject: Rational::zero(),
            var_supports: VarSupport::empty(n),
        }
    }

    pub(crate) fn marginalize_out(&self, var: Var) -> ResidualBound {
        let mut var_supports = self.var_supports.clone();
        if !var_supports[var].is_empty() {
            var_supports.set(var, SupportSet::zero());
        }
        ResidualBound {
            lower: self.lower.marginalize_out(var),
            reject: self.reject.clone(),
            var_supports,
        }
    }

    pub fn marginal(&self, var: Var) -> Self {
        let mut result = self.clone();
        for v in 0..result.var_supports.num_vars() {
            if Var(v) != var {
                result = result.marginalize_out(Var(v));
            }
        }
        result
    }

    pub(crate) fn var_count(&self) -> usize {
        self.lower.masses.ndim()
    }

    pub(crate) fn add_reject(mut self, reject: Rational) -> Self {
        self.reject += reject;
        self
    }

    pub fn residual(&self) -> Rational {
        Rational::one() - self.lower.total_mass() - self.reject.clone()
    }
}

impl std::ops::AddAssign for ResidualBound {
    fn add_assign(&mut self, rhs: Self) {
        self.lower += &rhs.lower;
        self.reject += rhs.reject;
        self.var_supports = self.var_supports.join(&rhs.var_supports);
    }
}

impl std::ops::Add for ResidualBound {
    type Output = ResidualBound;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::fmt::Display for ResidualBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}↯ + {}", self.reject, self.lower)
    }
}
