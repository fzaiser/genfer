// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of Descent Model trait for IPOPT
//!
//! This contains the IPOPT bindings and implementation of `Model` in the
//! `descent` crate.
//!
//! # Examples
//!
//! Using fixed expressions:
//!
//! ```
//! #![feature(proc_macro_hygiene)]
//!
//! use descent::model::Model;
//! use descent_ipopt::IpoptModel;
//! use descent_macro::expr;
//!
//! let mut m = IpoptModel::new();
//!
//! let x = m.add_var(-10.0, 10.0, 0.0);
//! let y = m.add_var(std::f64::NEG_INFINITY, std::f64::INFINITY, 0.0);
//! m.set_obj(expr!(2.0 * y; y));
//! m.add_con(expr!(y - x * x + x; x, y), 0.0, std::f64::INFINITY);
//!
//! let (stat, sol) = m.solve().unwrap();
//!
//! assert!(stat == descent::model::SolutionStatus::Solved);
//! assert!((sol.obj_val - (-0.5)).abs() < 1e-5);
//! assert!((sol.var(x) - 0.5).abs() < 1e-5);
//! assert!((sol.var(y) - (-0.25)).abs() < 1e-5);
//! ```
//!
//! Using dynamic expressions:
//!
//! ```
//! use descent::model::Model;
//! use descent_ipopt::IpoptModel;
//!
//! let mut m = IpoptModel::new();
//!
//! let x = m.add_var(-10.0, 10.0, 0.0);
//! let y = m.add_var(std::f64::NEG_INFINITY, std::f64::INFINITY, 0.0);
//! m.set_obj(2.0 * y);
//! m.add_con(y - x * x + x, 0.0, std::f64::INFINITY);
//!
//! let (stat, sol) = m.solve().unwrap();
//!
//! assert!(stat == descent::model::SolutionStatus::Solved);
//! assert!((sol.obj_val - (-0.5)).abs() < 1e-5);
//! assert!((sol.var(x) - 0.5).abs() < 1e-5);
//! assert!((sol.var(y) - (-0.25)).abs() < 1e-5);
//! ```

mod ipopt;
mod sparsity;

use crate::ipopt::ApplicationReturnStatus;
use crate::sparsity::{HesSparsity, JacSparsity, Sparsity};
use descent::expr::dynam::WorkSpace;
use descent::expr::{Column, Expression, Retrieve};
use descent::expr::{Par, Var};
use descent::model;
use descent::model::{Con, Model, Solution, SolutionStatus};
use std::slice;

use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum Error {
    UnpreparedModel,
    /// Create function returned null pointer.
    CreateIpopt,
    /// Ipopt needs at least 1 variable to be constructed.
    NoVariables,
    /// Failed to set option.
    SettingOption,
    /// Ipopt problem missing.
    MissingIpoptProblem,
}

impl From<Error> for model::Error {
    fn from(err: Error) -> Self {
        model::Error::Solving {
            source: Box::new(err),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

struct Variable {
    lb: f64,
    ub: f64,
    init: f64,
}

struct Parameter {
    val: f64,
}

struct Constraint {
    expr: Expression,
    lb: f64,
    ub: f64,
}

struct Objective {
    expr: Expression,
}

struct ModelData {
    vars: Vec<Variable>,
    pars: Vec<Parameter>,
    cons: Vec<Constraint>,
    obj: Objective,
}

#[derive(Debug, Default)]
struct ModelCache {
    sparsity: Sparsity, // jacobian and hessian sparsity
    ws: WorkSpace,
    cons_const: Vec<Column>,
    obj_const: Column,
    cons: Vec<Column>,
    obj: Column,
    //f_count: (usize, usize),
    //f_grad_count: (usize, usize),
    //g_count: (usize, usize),
    //g_jac_count: (usize, usize),
    //l_hess_count: (usize, usize),
}

// Make sure don't implement copy or clone for this otherwise risk of double
// free.
struct IpoptProblem {
    prob: ipopt::IpoptProblem,
}

impl Drop for IpoptProblem {
    fn drop(&mut self) {
        unsafe { ipopt::FreeIpoptProblem(self.prob) };
    }
}

// While we can't copy or clone them, it should be safe to pass them between
// threads, assuming ipopt isn't sharing anything mutable between problems.
unsafe impl Send for IpoptProblem {}
unsafe impl Sync for IpoptProblem {}

/// IPOPT model / solver.
pub struct IpoptModel {
    model: ModelData,
    cache: Option<ModelCache>,
    prob: Option<IpoptProblem>,
    prepared: bool,                                         // problem prepared
    pub last_ipopt_status: Option<ApplicationReturnStatus>, // status of last solve
}

impl Default for IpoptModel {
    fn default() -> Self {
        IpoptModel {
            model: ModelData {
                vars: Vec::new(),
                pars: Vec::new(),
                cons: Vec::new(),
                obj: Objective {
                    expr: descent::expr::dynam::Expr::from(0.0).into(),
                },
            },
            cache: None,
            prob: None,
            prepared: false,
            last_ipopt_status: None,
        }
    }
}

impl IpoptModel {
    pub fn new() -> Self {
        Self::default()
    }

    fn prepare(&mut self) -> Result<()> {
        // Have a problem if Expr is empty.  Don't know how to easy enforce
        // a non-empty Expr.  Could verify them, but then makes interface
        // clumbsy.  Could panic late like here.
        // Hrmm should possibly verify at the prepare phase.  Then return solve
        // error if things go bad.
        // Other option is to check as added to model (so before ExprInfo is
        // called).  Ideally should design interface/operations on ExprInfo
        // so that an empty/invalid value is not easily created/possible.
        if self.prepared && self.cache.is_some() && self.prob.is_some() {
            return Ok(()); // If still valid don't prepare again
        }

        // Ipopt cannot handle being created without any variables
        if self.model.vars.is_empty() {
            Err(Error::NoVariables)?;
        }

        let mut x_lb: Vec<f64> = Vec::new();
        let mut x_ub: Vec<f64> = Vec::new();
        for v in &self.model.vars {
            x_lb.push(v.lb);
            x_ub.push(v.ub);
        }

        let mut g_lb: Vec<f64> = Vec::new();
        let mut g_ub: Vec<f64> = Vec::new();
        let mut sparsity = Sparsity::new();

        sparsity.add_obj(&self.model.obj.expr);

        for c in self.model.cons.iter() {
            g_lb.push(c.lb);
            g_ub.push(c.ub);
            sparsity.add_con(&c.expr);
        }

        let nvars = self.model.vars.len();
        let ncons = self.model.cons.len();
        let nele_jac = sparsity.jac_len();
        let nele_hes = sparsity.hes_len();

        // x_lb, x_ub, g_lb, g_ub are copied internally so don't need to keep
        let prob = unsafe {
            ipopt::CreateIpoptProblem(
                nvars as i32,
                x_lb.as_ptr(),
                x_ub.as_ptr(),
                ncons as i32,
                g_lb.as_ptr(),
                g_ub.as_ptr(),
                nele_jac as i32,
                nele_hes as i32,
                0, // C-style indexing
                f,
                g,
                f_grad,
                g_jac,
                l_hess,
            )
        };
        if prob.is_null() {
            Err(Error::CreateIpopt)?;
        }

        // From code always returns true
        // For some reason getting incorrect/corrupt callback data
        // Don't need anymore because using new_x
        //unsafe { ipopt::SetIntermediateCallback(prob, intermediate) };

        let mut cache = ModelCache {
            sparsity,
            ..Default::default()
        };
        cache.cons.resize(ncons, Column::new());

        // Need to allocate memory for fixed expressions.
        // Don't need this for ExprDyn as done dynamically, but doing anyway.
        cache.obj.der1.resize(self.model.obj.expr.d1_len(), 0.0);
        cache.obj.der2.resize(self.model.obj.expr.d2_len(), 0.0);
        for (cc, c) in cache.cons.iter_mut().zip(self.model.cons.iter()) {
            cc.der1.resize(c.expr.d1_len(), 0.0);
            cc.der2.resize(c.expr.d2_len(), 0.0);
        }

        self.prob = Some(IpoptProblem { prob });
        self.cache = Some(cache);
        self.prepared = true;
        Ok(())
    }

    /// Set an IPOPT string option.
    ///
    /// Options can only be set once model is prepared. They will be lost if
    /// the model is modified.
    pub fn set_str_option(&mut self, key: &str, val: &str) -> bool {
        if self.prepare().is_err() {
            false
        } else if let Some(prob) = self.prob.as_mut() {
            let key_c = std::ffi::CString::new(key).unwrap();
            let val_c = std::ffi::CString::new(val).unwrap();
            unsafe { ipopt::AddIpoptStrOption(prob.prob, key_c.as_ptr(), val_c.as_ptr()) != 0 }
        } else {
            false
        }
    }

    /// Set an IPOPT numerical option.
    ///
    /// Options can only be set once model is prepared. They will be lost if
    /// the model is modified.
    pub fn set_num_option(&mut self, key: &str, val: f64) -> bool {
        if self.prepare().is_err() {
            false
        } else if let Some(prob) = self.prob.as_mut() {
            let key_c = std::ffi::CString::new(key).unwrap();
            unsafe { ipopt::AddIpoptNumOption(prob.prob, key_c.as_ptr(), val) != 0 }
        } else {
            false
        }
    }

    /// Set an IPOPT integer option.
    ///
    /// Options can only be set once model is prepared. They will be lost if
    /// the model is modified.
    pub fn set_int_option(&mut self, key: &str, val: i32) -> bool {
        if self.prepare().is_err() {
            false
        } else if let Some(prob) = self.prob.as_mut() {
            let key_c = std::ffi::CString::new(key).unwrap();
            unsafe { ipopt::AddIpoptIntOption(prob.prob, key_c.as_ptr(), val) != 0 }
        } else {
            false
        }
    }

    /// Silence the IPOPT default output.
    ///
    /// Options can only be set once model is prepared. They will be lost if
    /// the model is modified.
    pub fn silence(&mut self) -> bool {
        self.set_str_option("sb", "yes") && self.set_int_option("print_level", 0)
    }

    fn form_init_solution(&self, sol: &mut Solution) {
        // If no missing initial values, pull from variable
        let nvar_store = sol.store.vars.len();
        sol.store
            .vars
            .extend(self.model.vars.iter().skip(nvar_store).map(|x| x.init));
        sol.store.vars.resize(self.model.vars.len(), 0.0); // if need to shrink
                                                           // Always redo parameters
        sol.store.pars.clear();
        for p in &self.model.pars {
            sol.store.pars.push(p.val);
        }
        // Buffer rest with zeros
        sol.con_mult.resize(self.model.cons.len(), 0.0);
        sol.var_lb_mult.resize(self.model.vars.len(), 0.0);
        sol.var_ub_mult.resize(self.model.vars.len(), 0.0);
    }

    // Should only be called after prepare
    fn ipopt_solve(&mut self, mut sol: Solution) -> Result<(SolutionStatus, Solution)> {
        self.last_ipopt_status = None;
        self.form_init_solution(&mut sol);

        if let (&mut Some(ref mut cache), &Some(ref prob)) = (&mut self.cache, &self.prob) {
            // Calculating constant values in ExprDyn expressions.
            // Passing it the solution store (in theory var values should
            // not affect the values).
            cache.cons_const.clear();
            for c in &self.model.cons {
                cache
                    .cons_const
                    .push(evaluate_const(&c.expr, &sol.store, &mut cache.ws));
            }

            cache.obj_const = evaluate_const(&self.model.obj.expr, &sol.store, &mut cache.ws);

            let mut cb_data = IpoptCBData {
                model: &self.model,
                cache,
                pars: &sol.store.pars,
            };

            let cb_data_ptr = &mut cb_data as *mut _ as ipopt::UserDataPtr;

            // This and others might throw an exception. How would we catch?
            let ipopt_status = unsafe {
                ipopt::IpoptSolve(
                    prob.prob,
                    sol.store.vars.as_mut_ptr(),
                    std::ptr::null_mut(), // can calc ourselves
                    &mut sol.obj_val,
                    sol.con_mult.as_mut_ptr(),
                    sol.var_lb_mult.as_mut_ptr(),
                    sol.var_ub_mult.as_mut_ptr(),
                    cb_data_ptr,
                )
            };
            //println!("Counts: {:?} {:?} {:?} {:?} {:?}",
            //         cache.f_count,
            //         cache.f_grad_count,
            //         cache.g_count,
            //         cache.g_jac_count,
            //         cache.l_hess_count
            //         );

            // Saving status so can access from IpoptModel struct.
            self.last_ipopt_status = Some(ipopt_status.clone());
            Ok((ipopt_status.into(), sol))
        } else {
            Err(Error::UnpreparedModel)
        }
    }

    pub fn sparsity(&mut self) -> Option<(JacSparsity, HesSparsity)> {
        if self.prepare().is_err() {
            None
        } else if let Some(cache) = &self.cache {
            Some((cache.sparsity.jac_sp.clone(), cache.sparsity.hes_sp.clone()))
        } else {
            None
        }
    }
}

impl From<ApplicationReturnStatus> for SolutionStatus {
    fn from(ars: ApplicationReturnStatus) -> Self {
        use ApplicationReturnStatus as ARS;
        use SolutionStatus as SS;
        match ars {
            ARS::SolveSucceeded | ARS::SolvedToAcceptableLevel => SS::Solved,
            ARS::InfeasibleProblemDetected => SS::Infeasible,
            _ => SS::Other,
        }
    }
}

struct IpoptCBData<'a> {
    model: &'a ModelData,
    cache: &'a mut ModelCache,
    pars: &'a Vec<f64>,
}

impl Model for IpoptModel {
    fn add_var(&mut self, lb: f64, ub: f64, init: f64) -> Var {
        self.prepared = false;
        let id = self.model.vars.len();
        self.model.vars.push(Variable { lb, ub, init });
        Var(id)
    }

    fn add_par(&mut self, val: f64) -> Par {
        self.prepared = false;
        let id = self.model.pars.len();
        self.model.pars.push(Parameter { val });
        Par(id)
    }

    fn add_con<E: Into<Expression>>(&mut self, expr: E, lb: f64, ub: f64) -> Con {
        self.prepared = false;
        let id = self.model.cons.len();
        self.model.cons.push(Constraint {
            expr: expr.into(),
            lb,
            ub,
        });
        Con(id)
    }

    fn set_obj<E: Into<Expression>>(&mut self, expr: E) {
        self.prepared = false;
        self.model.obj = Objective { expr: expr.into() };
    }

    /// Set parameter to value.
    ///
    /// # Panics
    ///
    /// Expect a panic if parameter not in model.
    fn set_par(&mut self, par: Par, val: f64) {
        self.model.pars[par.0].val = val;
    }

    /// Change the variable lower bound.
    ///
    /// # Panics
    ///
    /// Expect a panic if variable not in model.
    fn set_lb(&mut self, var: Var, lb: f64) {
        self.model.vars[var.0].lb = lb;
    }

    /// Change the variable upper bound.
    ///
    /// # Panics
    ///
    /// Expect a panic if variable not in model.
    fn set_ub(&mut self, var: Var, ub: f64) {
        self.model.vars[var.0].ub = ub;
    }

    /// Set variable initial value.
    ///
    /// # Panics
    ///
    /// Expect a panic if variable not in model.
    fn set_init(&mut self, var: Var, init: f64) {
        self.model.vars[var.0].init = init;
    }

    /// Solve the model.
    ///
    /// # Panics
    ///
    /// Expression supplied to the model should only contain valid `Var` and
    /// `Par` values, i.e. those that were returned by the `add_var` / `add_par`
    /// methods by this model. Panic can occur here if not.
    fn solve(&mut self) -> model::Result<(SolutionStatus, Solution)> {
        self.prepare()?;
        let sol = Solution::new();
        Ok(self.ipopt_solve(sol)?) // Have prepared so cannot fail
    }

    /// Solve the model using the supplied solution as a warm start.
    ///
    /// # Panics
    ///
    /// Expression supplied to the model should only contain valid `Var` and
    /// `Par` values, i.e. those that were returned by the `add_var` / `add_par`
    /// methods by this model. Panic can occur here if not.
    fn warm_solve(&mut self, sol: Solution) -> model::Result<(SolutionStatus, Solution)> {
        self.prepare()?;
        // Should set up warm start stuff
        Ok(self.ipopt_solve(sol)?) // Have prepared so cannot fail
    }
}

struct Store<'a> {
    vars: &'a [ipopt::Number], // reference values provided in IPOPT callbacks
    pars: &'a [f64],           // values stored in model
}

impl<'a> Retrieve for Store<'a> {
    fn var(&self, v: Var) -> f64 {
        self.vars[v.0]
    }

    fn par(&self, p: Par) -> f64 {
        self.pars[p.0]
    }
}

/// Produces const evaluation of an expression.
fn evaluate_const<R>(expr: &Expression, store: &R, mut ws: &mut WorkSpace) -> Column
where
    R: Retrieve,
{
    match expr {
        Expression::ExprDyn(e) => e.auto_const(store, &mut ws),
        Expression::ExprDynSum(es) => {
            let mut col = Column::new();
            for e in es {
                col.sum_concat(e.auto_const(store, &mut ws));
            }
            col
        }
        _ => {
            // Others don't support constant evaluations.
            Column::new()
        }
    }
}

fn evaluate(expr: &Expression, store: &Store, col: &mut Column, mut ws: &mut WorkSpace) {
    match expr {
        Expression::ExprFix(expr) => {
            col.val = (expr.all)(store.vars, store.pars, &mut col.der1, &mut col.der2);
        }
        Expression::ExprFixSum(es) => {
            col.val = 0.0;
            let mut der1_off = 0;
            let mut der2_off = 0;
            for expr in es {
                let d1_len = expr.d1_sparsity.len();
                let d2_len = expr.d2_sparsity.len();
                col.val += (expr.all)(
                    store.vars,
                    store.pars,
                    &mut col.der1[der1_off..der1_off + d1_len],
                    &mut col.der2[der2_off..der2_off + d2_len],
                );
                der1_off += d1_len;
                der2_off += d2_len;
            }
        }
        Expression::ExprDyn(e) => {
            *col = e.auto_dynam(store, &mut ws);
        }
        Expression::ExprDynSum(es) => {
            col.val = 0.0;
            col.der1.clear();
            col.der2.clear();
            for e in es {
                col.sum_concat(e.auto_dynam(store, &mut ws));
            }
        }
    }
}

/// Need to initialise the columns to correct lengths for static expr's
fn evaluate_obj(cb_data: &mut IpoptCBData, store: &Store) {
    evaluate(
        &cb_data.model.obj.expr,
        &store,
        &mut cb_data.cache.obj,
        &mut cb_data.cache.ws,
    );
}

/// Need to initialise the columns to correct lengths for static expr's
fn evaluate_cons(cb_data: &mut IpoptCBData, store: &Store) {
    for (mut cc, c) in cb_data.cache.cons.iter_mut().zip(cb_data.model.cons.iter()) {
        evaluate(&c.expr, &store, &mut cc, &mut cb_data.cache.ws);
    }
}

// Would be possibly to tune these functions by lazily calculating the bits
// that are required for each new_x as they are requested. Would need to track
// a state in the cache that gets reset each time a new_x is observed.
// This could maybe save around 20% of jacobian and even more hessian calcs.
// There will be a balance, as the evaluation sometimes calculate other values
// as a by-product (e.g., the full_fwd call).
// For example problem (non-sparsity function calls, those that had new_x):
// f (99, 3) f_grad (73, 1) g (99, 95) g_jac (80, 1) l_hess (72, 0)
//
// For f_grad looks like one first call values are not saved, but after
// that they are. g_jac doesn't have same issue. l_hess is completely over the
// place, (scaling going on?). Might use some of the above to avoid copying
// constant values more than once.

extern "C" fn f(
    n: ipopt::Index,
    x: *const ipopt::Number,
    new_x: ipopt::Bool,
    obj_value: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr,
) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe { &mut *(user_data as *mut IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }

    //cb_data.cache.f_count.0 += 1;
    if new_x == 1 {
        //cb_data.cache.f_count.1 += 1;
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: cb_data.pars.as_slice(),
        };

        evaluate_obj(cb_data, &store);
        evaluate_cons(cb_data, &store);
    }

    let value = unsafe { &mut *obj_value };
    *value = cb_data.cache.obj.val;
    1
}

extern "C" fn f_grad(
    n: ipopt::Index,
    x: *const ipopt::Number,
    new_x: ipopt::Bool,
    grad_f: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr,
) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe { &mut *(user_data as *mut IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }

    //cb_data.cache.f_grad_count.0 += 1;
    if new_x == 1 {
        //cb_data.cache.f_grad_count.1 += 1;
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: cb_data.pars,
        };

        evaluate_obj(cb_data, &store);
        evaluate_cons(cb_data, &store);
    }

    let values = unsafe { slice::from_raw_parts_mut(grad_f, n as usize) };

    for v in values.iter_mut() {
        *v = 0.0;
    }

    // f_grad expects variables in order
    for (i, v) in cb_data.model.obj.expr.lin_iter().enumerate() {
        values[v] += cb_data.cache.obj_const.der1[i];
    }
    for (i, v) in cb_data.model.obj.expr.nlin_iter().enumerate() {
        values[v] += cb_data.cache.obj.der1[i];
    }
    1
}

extern "C" fn g(
    n: ipopt::Index,
    x: *const ipopt::Number,
    new_x: ipopt::Bool,
    m: ipopt::Index,
    g: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr,
) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe { &mut *(user_data as *mut IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }

    //cb_data.cache.g_count.0 += 1;
    if new_x == 1 {
        //cb_data.cache.g_count.1 += 1;
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: cb_data.pars,
        };

        evaluate_obj(cb_data, &store);
        evaluate_cons(cb_data, &store);
    }

    let values = unsafe { slice::from_raw_parts_mut(g, m as usize) };

    for (i, col) in cb_data.cache.cons.iter().enumerate() {
        values[i] = col.val;
    }
    1
}

extern "C" fn g_jac(
    n: ipopt::Index,
    x: *const ipopt::Number,
    new_x: ipopt::Bool,
    m: ipopt::Index,
    nele_jac: ipopt::Index,
    i_row: *mut ipopt::Index,
    j_col: *mut ipopt::Index,
    vals: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr,
) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe { &mut *(user_data as *mut IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }
    if nele_jac != cb_data.cache.sparsity.jac_len() as i32 {
        return 0;
    }

    if vals.is_null() {
        // Set sparsity
        let row = unsafe { slice::from_raw_parts_mut(i_row, nele_jac as usize) };
        let col = unsafe { slice::from_raw_parts_mut(j_col, nele_jac as usize) };
        for (&(cid, vid), i) in cb_data.cache.sparsity.jac_sp.iter() {
            row[*i] = cid as i32;
            col[*i] = vid as i32;
        }
    } else {
        // Set values
        //cb_data.cache.g_jac_count.0 += 1;
        if new_x == 1 {
            //cb_data.cache.g_jac_count.1 += 1;
            let store = Store {
                vars: unsafe { slice::from_raw_parts(x, n as usize) },
                pars: cb_data.pars,
            };

            evaluate_obj(cb_data, &store);
            evaluate_cons(cb_data, &store);
        }

        let values = unsafe { slice::from_raw_parts_mut(vals, nele_jac as usize) };

        for v in values.iter_mut() {
            *v = 0.0;
        }

        let cache = &cb_data.cache;
        for ((col, col_const), inds) in cache
            .cons
            .iter()
            .zip(cache.cons_const.iter())
            .zip(cache.sparsity.jac_cons_inds.iter())
        {
            for (i, val) in inds
                .iter()
                .zip(col_const.der1.iter().chain(col.der1.iter()))
            {
                values[*i] += *val;
            }
        }
    }
    1
}

extern "C" fn l_hess(
    n: ipopt::Index,
    x: *const ipopt::Number,
    new_x: ipopt::Bool,
    obj_factor: ipopt::Number,
    m: ipopt::Index,
    lambda: *const ipopt::Number,
    _new_lambda: ipopt::Bool,
    nele_hes: ipopt::Index,
    i_row: *mut ipopt::Index,
    j_col: *mut ipopt::Index,
    vals: *mut ipopt::Number,
    user_data: ipopt::UserDataPtr,
) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe { &mut *(user_data as *mut IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }
    if nele_hes != cb_data.cache.sparsity.hes_len() as i32 {
        return 0;
    }

    if vals.is_null() {
        // Set sparsity
        let row = unsafe { slice::from_raw_parts_mut(i_row, nele_hes as usize) };
        let col = unsafe { slice::from_raw_parts_mut(j_col, nele_hes as usize) };
        for (vids, &ind) in &cb_data.cache.sparsity.hes_sp {
            row[ind] = vids.0 as i32;
            col[ind] = vids.1 as i32;
        }
    } else {
        // Set values
        //cb_data.cache.l_hess_count.0 += 1;
        if new_x == 1 {
            //cb_data.cache.l_hess_count.1 += 1;
            let store = Store {
                vars: unsafe { slice::from_raw_parts(x, n as usize) },
                pars: cb_data.pars,
            };

            evaluate_obj(cb_data, &store);
            evaluate_cons(cb_data, &store);
        }

        let lam = unsafe { slice::from_raw_parts(lambda, m as usize) };
        let values = unsafe { slice::from_raw_parts_mut(vals, nele_hes as usize) };

        for v in values.iter_mut() {
            *v = 0.0;
        }

        let cache = &cb_data.cache;
        for (i, val) in cache
            .sparsity
            .hes_obj_inds
            .iter()
            .zip(cache.obj_const.der2.iter().chain(cache.obj.der2.iter()))
        {
            values[*i] += obj_factor * val;
        }

        for (((col, col_const), inds), l) in cache
            .cons
            .iter()
            .zip(cache.cons_const.iter())
            .zip(cache.sparsity.hes_cons_inds.iter())
            .zip(lam.iter())
        {
            for (i, val) in inds
                .iter()
                .zip(col_const.der2.iter().chain(col.der2.iter()))
            {
                values[*i] += l * val;
            }
        }
    }
    1
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;
    use descent::expr::dynam::NumOps;
    use std::f64;
    #[test]
    fn univar_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        m.set_obj(x * x);
        assert!(m.silence());
        let (stat, sol) = m.solve().unwrap();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!((sol.var(x) - 1.0).abs() < 1e-6);
        assert!((sol.obj_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn multivar_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(x * x + y * y + x * y);
        m.silence();
        let (stat, sol) = m.solve().unwrap();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!((sol.var(x) - 1.0).abs() < 1e-6);
        assert!((sol.var(y) + 0.5).abs() < 1e-6);
        assert!((sol.obj_val - 0.75).abs() < 1e-6);
    }

    #[test]
    fn equality_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(x * x + y * y + x * y);
        m.add_con(x + y, 0.75, 0.75);
        m.silence();
        let (stat, sol) = m.solve().unwrap();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!((sol.var(x) - 1.0).abs() < 1e-6);
        assert!((sol.var(y) + 0.25).abs() < 1e-6);
        assert!((sol.obj_val - 0.8125).abs() < 1e-6);
    }

    #[test]
    fn inequality_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(x * x + y * y + x * y);
        m.add_con(x + y, 0.25, 0.40);
        m.silence();
        let (stat, sol) = m.solve().unwrap();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!((sol.var(x) - 1.0).abs() < 1e-6);
        assert!((sol.var(y) + 0.6).abs() < 1e-6);
        assert!((sol.obj_val - 0.76).abs() < 1e-6);
    }

    #[test]
    fn quad_constraint_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(-10.0, 10.0, 0.0);
        let y = m.add_var(f64::NEG_INFINITY, f64::INFINITY, 0.0);
        m.set_obj(2.0 * y);
        m.add_con(y - x * x + x, 0.0, f64::INFINITY);
        m.silence();
        let (stat, sol) = m.solve().unwrap();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!((sol.var(x) - 0.5).abs() < 1e-5);
        assert!((sol.var(y) + 0.25).abs() < 1e-5);
        assert!((sol.obj_val + 0.5).abs() < 1e-4);
    }

    #[test]
    fn separation() {
        let mut m = IpoptModel::new();
        let mut xs = Vec::new();
        for _i in 0..6 {
            xs.push(m.add_var(-10.0, 10.0, 0.0));
        }

        let mut obj = descent::expr::dynam::Expr::from(0.0);
        for &x in &xs {
            obj = obj + (x - 1.0).powi(2);
        }
        m.set_obj(obj);
        m.silence();
        let (_, sol) = m.solve().unwrap();
        assert!(sol.obj_val.abs() < 1e-5);
        for &x in &xs {
            assert!((sol.var(x) - 1.0).abs() < 1e-5);
        }
    }

    #[bench]
    fn solve_larger(b: &mut test::Bencher) {
        let n = 5;
        let mut m = IpoptModel::new();
        let mut xs = Vec::new();
        for _i in 0..n {
            xs.push(m.add_var(-1.5, 0.0, -0.5));
        }
        let mut obj = descent::expr::dynam::Expr::from(0.0);
        for &x in &xs {
            obj = obj + (x - 1.0).powi(2);
        }
        m.set_obj(obj);
        for i in 0..(n - 2) {
            let a = ((i + 2) as f64) / (n as f64);
            let expr = (xs[i + 1].powi(2) + 1.5 * xs[i + 1] - a) * xs[i + 2].cos() - xs[i];
            m.add_con(expr, 0.0, 0.0);
        }
        m.silence();
        b.iter(|| {
            m.solve().unwrap();
        });
    }
}
