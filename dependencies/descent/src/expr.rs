// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Expression, variable and parameter modelling.
//!
//! General types used in expressions. See sub-modules for details of working
//! with either fixed (compile-time) or dynamically (run-time) constructed
//! expressions.

pub mod dynam;
pub mod fixed;

use self::dynam::{ExprDyn, ExprDynSum};
use self::fixed::{ExprFix, ExprFixSum};

/// Identifier used for variables and parameters.
pub type ID = usize;

/// Variable identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Var(pub ID);

/// Parameter identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Par(pub ID);

/// Retrieve current values of variables and parameters.
///
/// Expect a panic if requested id not available for whatever reason.
pub trait Retrieve {
    fn var(&self, v: Var) -> f64;
    fn par(&self, p: Par) -> f64;
}

/// Extract values of constant valued items given a slice of paramter values.
/// 
/// This is a helper trait for procedural macro and otherwise generally should
/// not be used.
pub trait Extract {
    fn extract(&self, pars: &[f64]) -> f64;
}

impl Extract for f64 {
    fn extract(&self, _pars: &[f64]) -> f64 {
        *self
    }
}

//impl<T: std::ops::Deref<Target=f64>> Extract for T {
//    fn get_noise(&self, _pars: &[f64]) -> f64 {
//        *self.deref()
//    }
//}

impl Extract for Par {
    /// Expect a panic if requested id not available for whatever reason.
    fn extract(&self, pars: &[f64]) -> f64 {
        pars[self.0]
    }
}

/// Storage for variable and parameter values.
#[derive(Debug, Clone, Default)]
pub struct Store {
    pub vars: Vec<f64>,
    pub pars: Vec<f64>,
}

impl Store {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_var(&mut self, value: f64) -> Var {
        let id = self.vars.len();
        self.vars.push(value);
        Var(id)
    }

    pub fn add_par(&mut self, value: f64) -> Par {
        let id = self.pars.len();
        self.pars.push(value);
        Par(id)
    }
}

impl Retrieve for Store {
    fn var(&self, v: Var) -> f64 {
        self.vars[v.0]
    }

    fn par(&self, p: Par) -> f64 {
        self.pars[p.0]
    }
}

/// Most general representation of an expression.
///
/// This wraps up fixed and dynamic types of expressions and their respective
/// summation forms. The user typically doesn't directly need to care about
/// this type unless they want a way to easily store different types of
/// expressions in the same collection. See the [fixed](fixed/index.html) or
/// [dynam](dynam/index.html) sub-modules for details about constructing and
/// working with expressions.
///
/// Only a few methods are implemented directly for this type. More could be
/// developed to provide a cleaner interface to solvers.
pub enum Expression {
    ExprFix(ExprFix),
    ExprFixSum(ExprFixSum),
    ExprDyn(ExprDyn),
    ExprDynSum(ExprDynSum),
}

impl Expression {
    /// Iterate over variables that we want to treat as linear.
    ///
    /// Implemtation assumes all varialbes in `ExprFix` are non-linear,
    /// because it will have minimal extra overhead doing so.
    pub fn lin_iter<'a>(&'a self) -> Box<dyn Iterator<Item = ID> + 'a> {
        match self {
            Expression::ExprFix(_) => Box::new(std::iter::empty()),
            Expression::ExprFixSum(_) => Box::new(std::iter::empty()),
            Expression::ExprDyn(e) => Box::new(e.info.lin.iter().cloned()),
            Expression::ExprDynSum(es) => {
                Box::new(es.iter().map(|e| e.info.lin.iter().cloned()).flatten())
            }
        }
    }

    /// Iterate over variables that we want to treat as non-linear.
    ///
    /// Can contain variables at that are actually linear if we don't want
    /// special treatment for them.
    pub fn nlin_iter<'a>(&'a self) -> Box<dyn Iterator<Item = ID> + 'a> {
        match self {
            Expression::ExprFix(e) => Box::new(e.d1_sparsity.iter().map(|Var(v)| *v)),
            Expression::ExprFixSum(es) => Box::new(
                es.iter()
                    .map(|e| e.d1_sparsity.iter().map(|Var(v)| *v))
                    .flatten(),
            ),
            Expression::ExprDyn(e) => Box::new(e.info.nlin.iter().cloned()),
            Expression::ExprDynSum(es) => {
                Box::new(es.iter().map(|e| e.info.nlin.iter().cloned()).flatten())
            }
        }
    }

    /// Number of entries calculated for first derivative.
    ///
    /// These will typically be the non-zeros but not necessarily. Also, one
    /// variable could be accounted for twice if one of the summation types.
    pub fn d1_len(&self) -> usize {
        match self {
            Expression::ExprFix(e) => e.d1_sparsity.len(),
            Expression::ExprFixSum(es) => {
                let mut count = 0;
                for e in es {
                    count += e.d1_sparsity.len();
                }
                count
            }
            Expression::ExprDyn(e) => e.info.lin.len() + e.info.nlin.len(),
            Expression::ExprDynSum(es) => {
                let mut count = 0;
                for e in es {
                    count += e.info.lin.len() + e.info.nlin.len();
                }
                count
            }
        }
    }

    /// Number of entries calculated for second derivative.
    ///
    /// These will typically be the non-zeros but not necessarily. Also, pairs
    /// of variables could be accounted for twice if one of the summation types.
    pub fn d2_len(&self) -> usize {
        match self {
            Expression::ExprFix(e) => e.d2_sparsity.len(),
            Expression::ExprFixSum(es) => {
                let mut count = 0;
                for e in es {
                    count += e.d2_sparsity.len();
                }
                count
            }
            Expression::ExprDyn(e) => e.info.quad.len() + e.info.nquad.len(),
            Expression::ExprDynSum(es) => {
                let mut count = 0;
                for e in es {
                    count += e.info.quad.len() + e.info.nquad.len();
                }
                count
            }
        }
    }
}

/// Order second derivative pairs.
///
/// Should fill out bottom left of Hessian with this ordering.
pub fn order<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b {
        (b, a)
    } else {
        (a, b)
    }
}

/// Storage for evaluation of an expression.
#[derive(Debug, Clone, Default)]
pub struct Column {
    pub val: f64,
    pub der1: Vec<f64>,
    pub der2: Vec<f64>,
}

impl Column {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Column {
    pub fn sum_concat(&mut self, other: Column) {
        self.val += other.val;
        self.der1.extend(other.der1.into_iter());
        self.der2.extend(other.der2.into_iter());
    }
}
