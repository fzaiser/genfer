// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Model interface for solvers.

use crate::expr::{Expression, Par, Retrieve, Store, Var, ID};

use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum Error {
    Solving { source: Box<dyn std::error::Error> },
}

pub type Result<T> = std::result::Result<T, Error>;

//pub enum VarType {
//    Continuous,
//    Integer,
//    Binary,
//}

/// Constraint identifier.
#[derive(Debug, Clone, Copy)]
pub struct Con(pub ID);

/// Interface for a mathematical program with continuous variables.
///
/// # Panics
///
/// Expect a panic, either our of bounds or otherwise, if a variable or
/// parameter is used in this model that was not directly returned by the
/// `add_var` or `add_par` methods. Check individual implementations of trait
/// for details of when these panics could occur.
pub trait Model {
    /// Add variable to model with lower / upper bounds and initial value.
    fn add_var(&mut self, lb: f64, ub: f64, init: f64) -> Var;
    /// Add parameter to model with starting value.
    fn add_par(&mut self, val: f64) -> Par;
    /// Add a constraint to the model with lower and upper bounds.
    ///
    /// To have no lower / upper bounds set them to `std::f64::NEG_INFINITY` /
    /// `std::f64::INFINITY` respectively.
    fn add_con<E: Into<Expression>>(&mut self, expr: E, lb: f64, ub: f64) -> Con;
    /// Set objective of model.
    fn set_obj<E: Into<Expression>>(&mut self, expr: E);
    /// Change a parameter's value.
    fn set_par(&mut self, par: Par, val: f64);
    /// Change the variable lower bound.
    fn set_lb(&mut self, var: Var, lb: f64);
    /// Change the variable upper bound.
    fn set_ub(&mut self, var: Var, ub: f64);
    /// Change the initial value of a variable.
    fn set_init(&mut self, var: Var, init: f64);
    /// Solve the model.
    fn solve(&mut self) -> Result<(SolutionStatus, Solution)>;
    /// Solve the model using a previous solution as a warm start.
    fn warm_solve(&mut self, sol: Solution) -> Result<(SolutionStatus, Solution)>;
}

// Not used yet
//pub trait MIModel {
//    fn add_ivar(&mut self, lb: f64, ub: f64) -> Var;
//    fn add_bvar(&mut self, lb: f64, ub: f64) -> Var;
//}

/// Status of the solution.
#[derive(PartialEq, Debug)]
pub enum SolutionStatus {
    /// Problem successfully solved.
    Solved,
    /// Problem appears to be infeasible.
    Infeasible,
    /// Other currrently unhandled solver status returned.
    Other,
}

/// Data for a valid solution.
#[derive(Default)]
pub struct Solution {
    /// Objective value.
    pub obj_val: f64,
    /// Store of variable and parameter values.
    pub store: Store,
    /// Constraint Lagrange / KKT multipliers.
    pub con_mult: Vec<f64>,
    /// Variable lower bound KKT multipliers.
    pub var_lb_mult: Vec<f64>,
    /// Variable upper bound KKT multipliers.
    pub var_ub_mult: Vec<f64>,
}

impl Solution {
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the value of an expression using the solution.
    pub fn value(&self, expr: &Expression) -> f64 {
        match expr {
            Expression::ExprFix(e) => (e.f)(&self.store.vars, &self.store.pars),
            Expression::ExprFixSum(es) => {
                let mut val = 0.0;
                for e in es {
                    val += (e.f)(&self.store.vars, &self.store.pars);
                }
                val
            }
            Expression::ExprDyn(e) => {
                let mut ns = Vec::new();
                e.expr.eval(&self.store, &mut ns)
            }
            Expression::ExprDynSum(es) => {
                let mut ns = Vec::new();
                let mut val = 0.0;
                for e in es {
                    val += e.expr.eval(&self.store, &mut ns);
                }
                val
            }
        }
    }

    /// Get the value of variable for solution.
    pub fn var(&self, v: Var) -> f64 {
        self.store.var(v)
    }

    /// Get the constraint KKT / Lagrange multiplier.
    pub fn con_mult(&self, Con(cid): Con) -> f64 {
        self.con_mult[cid]
    }

    // Could write versions that take Expr, and try and match ops[0] to Var
    /// Get the variable lower bound constraint KKT / Lagrange multiplier.
    pub fn var_lb_mult(&self, Var(vid): Var) -> f64 {
        self.var_lb_mult[vid]
    }

    /// Get the variable upper bound constraint KKT / Lagrange multiplier.
    pub fn var_ub_mult(&self, Var(vid): Var) -> f64 {
        self.var_ub_mult[vid]
    }
}
