use crate::{ppl::Var, semantics::support::VarSupport, support::SupportSet};

use super::{Egd, FiniteDiscrete};

#[derive(Debug, Clone)]
pub struct GeometricBound {
    pub lower: FiniteDiscrete,
    pub upper: Egd,
    pub var_supports: VarSupport,
}

impl GeometricBound {
    pub(crate) fn zero(n: usize) -> GeometricBound {
        GeometricBound {
            lower: FiniteDiscrete::zero(n),
            upper: Egd::zero(n),
            var_supports: VarSupport::empty(n),
        }
    }

    pub(crate) fn marginalize_out(&self, var: Var) -> GeometricBound {
        let mut var_supports = self.var_supports.clone();
        if !var_supports[var].is_empty() {
            var_supports.set(var, SupportSet::zero());
        }
        GeometricBound {
            lower: self.lower.marginalize_out(var),
            upper: self.upper.marginalize_out(var),
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
}

impl std::fmt::Display for GeometricBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(support_vec) = self.var_supports.as_vec() {
            for (i, support) in support_vec.iter().enumerate() {
                writeln!(
                    f,
                    "Support of {var}: {support}",
                    var = Var(i),
                    support = support
                )?;
            }
        } else {
            writeln!(f, "Support: empty")?;
        }
        writeln!(f, "Lower bound:\n{bound}", bound = self.lower)?;
        write!(f, "Upper bound:\n{bound}", bound = self.upper)
    }
}
