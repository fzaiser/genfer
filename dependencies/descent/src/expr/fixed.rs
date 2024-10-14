// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Fixed expressions.
//!
//! Functions for calculating the first and second derivatives are generated
//! at compile-time for fixed expressions. The main type is
//! [ExprFix](struct.ExprFix.html) but instead of hand-writing its functions,
//! the user will typically want to auto-generate them from an expression using
//! the `expr!` procedural macro available in the `descent_macro` crate.
//!
//! Some limited run-time flexibility can be gained by using the ExprFixSum
//! type that represents a sum of ExprFix expressions.
//!
//! # Examples
//!
//! See the `descent_macro` crate for more comprehensive examples of the macro
//! use.
//!
//! Basic usage:
//!
//! ```ignore
//! #![feature(proc_macro_hygiene)] // need to turn on nightly feature
//! use descent::expr::{Var, Par};
//! use descent_macro::expr;
//! let x = Var(0);
//! let y = Var(1);
//! let p = Par(0);
//! let c = 1.0;
//! let e = expr!(p * x + y * y + c; x, y; p);
//! ```
//!
//! Making use of summation flexibility:
//!
//! ```ignore
//! #![feature(proc_macro_hygiene)] // need to turn on nightly feature
//! use descent::expr::Var;
//! use descent::expr::fixed::ExprFixSum;
//! let xs: Vec<Var> = (0..5).into_iter().map(|i| Var(i)).collect();
//!
//! let mut e = ExprFixSum::new();
//! for &x in &xs {
//!     e = e + expr!(x * x + x.sin(); x);
//! }
//! ```

use super::{Expression, Var};

// Should Use Rc instead of Box if we want to easily enable cloning.

/// Fixed expression with pointers to functions to evaluated the expression
/// and its first and second derivatives.
///
/// The input variable / parameter slices to these functions should be large
/// enough to include the indices of the vars / pars for the expression.
///
/// The sparsity may contain the same variable indice twice. The corresponding
/// first and second derivative values should be summed to get the overall
/// value for that variable.
pub struct ExprFix {
    /// Evaluate expression.
    pub f: Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>,
    /// Evaluate expression and its first and second derivatives in one go.
    ///
    /// Arguments are vars, pars, d1_out, d2_out
    pub all: Box<dyn Fn(&[f64], &[f64], &mut [f64], &mut [f64]) -> f64 + Send + Sync>,
    /// First derivate sparsity / order of outputs.
    pub d1_sparsity: Vec<Var>,
    /// Second derivate sparsity / order of outputs.
    pub d2_sparsity: Vec<(Var, Var)>,
}

impl From<ExprFix> for Expression {
    fn from(v: ExprFix) -> Self {
        Expression::ExprFix(v)
    }
}

/// Represents the sum of multiple fixed expressions.
///
/// This enables some more runtime flexibility without having to resort to a
/// `ExprDyn`.
pub type ExprFixSum = Vec<ExprFix>;

impl From<ExprFix> for ExprFixSum {
    fn from(v: ExprFix) -> Self {
        vec![v]
    }
}

impl From<ExprFixSum> for Expression {
    fn from(mut v: ExprFixSum) -> Self {
        if v.len() == 1 {
            Expression::ExprFix(v.pop().unwrap())
        } else {
            Expression::ExprFixSum(v)
        }
    }
}

impl std::ops::Add<ExprFix> for ExprFix {
    type Output = ExprFixSum;

    fn add(self, other: ExprFix) -> ExprFixSum {
        vec![self, other]
    }
}

impl std::ops::Add<ExprFixSum> for ExprFix {
    type Output = ExprFixSum;

    fn add(self, mut other: ExprFixSum) -> ExprFixSum {
        other.push(self);
        other
    }
}

impl std::ops::Add<ExprFix> for ExprFixSum {
    type Output = ExprFixSum;

    fn add(mut self, other: ExprFix) -> ExprFixSum {
        self.push(other);
        self
    }
}
