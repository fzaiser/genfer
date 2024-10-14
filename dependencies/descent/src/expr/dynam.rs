// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Dynamically constructed expressions.
//!
//! These expressions offer the greatest flexibility in how they can be
//! constructed at runtime. Forward and reverse automatic differentiation
//! techniques have been implemented for calculating expression first and
//! second derivatives. They are however much slower than the more direct
//! fixed expression types (in [fixed](../fixed/index.html) module) which allow
//! for greater compiler optimisation of the derivative calculations.
//!
//! # Examples
//!
//! See [Expr](struct.Expr.html) for more details.
//!
//! ```
//! use descent::expr::{Var, Par};
//! use descent::expr::dynam::NumOps; // required to bring in powi usage
//! let x = Var(0);
//! let y = Var(1);
//! let p = Par(0);
//!
//! let e = p * x - (5.0 * y.powi(2) + 4.0);
//! ```

use super::{Column, Expression, Par, Retrieve, Var, ID};
use std::collections::{HashMap, HashSet};
use std::ops::{Add, Mul, Sub};

/// Snapshot of a dynamically constructed expression `Expr`.
///
/// This wraps the final form of an `Expr` alongside expensive to compute
/// "sparsity" information about the expression.
pub struct ExprDyn {
    pub expr: Expr,
    pub info: ExprInfo,
}

impl From<ExprDyn> for Expression {
    fn from(expr: ExprDyn) -> Self {
        Expression::ExprDyn(expr)
    }
}

/// Represents the sum of multiple dynamic expressions.
///
/// Breaking up expressions into smaller sums of expressions can improve the
/// performance of AD techniques for large sums. If an `Expr` consists of
/// a large sum of terms, then when converted into an `Expression`, it will
/// automatically get converted into this type. Alternatively, this type can
/// be explicitly utilised as the addition operator is implement for it and
/// `Expr`:
///
/// ```
/// use descent::expr::Var;
/// use descent::expr::dynam::ExprDynSum;
/// let xs: Vec<Var> = (0..5).into_iter().map(|i| Var(i)).collect();
///
/// let mut e = ExprDynSum::new();
/// for &x in &xs {
///     e = e + 3.0 * x;
/// }
///
/// let mut e = ExprDynSum::new();
/// for &x in &xs {
///     e = e + x.into(); // variable needs to be converted to `Expr` first here
/// }
/// ```
pub type ExprDynSum = Vec<ExprDyn>;

impl From<Expr> for ExprDyn {
    fn from(expr: Expr) -> Self {
        let info = expr.get_info();
        ExprDyn { expr, info }
    }
}

impl From<ExprDynSum> for Expression {
    fn from(expr: ExprDynSum) -> Self {
        Expression::ExprDynSum(expr)
    }
}

impl std::ops::Add<ExprDynSum> for Expr {
    type Output = ExprDynSum;

    fn add(self, mut other: ExprDynSum) -> ExprDynSum {
        other.push(self.into());
        other
    }
}

impl std::ops::Add<Expr> for ExprDynSum {
    type Output = ExprDynSum;

    fn add(mut self, other: Expr) -> ExprDynSum {
        self.push(other.into());
        self
    }
}

/// Sparsity and other information about an `Expr` for guiding AD process.
///
/// Linear and quadratic respective first and second derivatives only need to
/// be computed once on parameter change. The `usize` pairs represent indices
/// into `nlin`. They are local variable mappings.
///
/// The contracts from `Degree` are expected to be held.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExprInfo {
    /// Constant first derivative
    pub lin: Vec<ID>,
    /// Non-constant first derivative
    pub nlin: Vec<ID>,
    /// Constant second derivative
    pub quad: Vec<(usize, usize)>,
    /// Non-constant second derivative
    pub nquad: Vec<(usize, usize)>,
    pub quad_list: Vec<Vec<ID>>,
    pub nquad_list: Vec<Vec<ID>>,
}

///// Only really care about this for second derivatives. Can just call reverse
///// method to get first derivatives instead of mapping them back.
//#[derive(Debug, Clone, Default, PartialEq)]
//pub struct SegInfo {
//    /// Segment position in Expr
//    pub pos: usize,
//    pub info: ExprInfo,
//    /// Map from local to parent position
//    pub quad_map: Vec<usize>,
//    /// Map from local to parent position
//    pub nquad_map: Vec<usize>,
//}

/// Maps for each `nlin` entry to IDs that pair with it in `quad`/`nquad`.
///
/// When everything is ordered, when traversed it will preserve original
/// `quad`/`nquad` orderings.
#[cfg(test)]
fn nlin_list(nlin: &[ID], sec: &[(usize, usize)]) -> Vec<Vec<ID>> {
    let mut vs = Vec::new();
    vs.resize(nlin.len(), Vec::new());
    for &(i, j) in sec {
        vs[i].push(nlin[j]);
    }
    vs
}

impl ExprInfo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Needs to be called before using some of the AD techniques.
    #[cfg(test)]
    pub(crate) fn set_lists(&mut self) {
        self.quad_list = nlin_list(&self.nlin, &self.quad);
        self.nquad_list = nlin_list(&self.nlin, &self.nquad);
    }
}

/// Reference to operand within `Expr`.
///
/// This references an operand of a operation within an `Expr`.  The operand
/// location `ORef` hops to the left.
type ORef = usize;

/// Operations for representing expressions.
///
/// These operations are designed to be stored and structured on a `Expr`.
/// They either have zero or more operands. For operations with 1 or more
/// operands, the first operand is implicitly the operation immediately to the
/// left on the `Expr`. For additional operands a relative position is given
/// using the `ORef` convention.
#[derive(Debug, Clone)]
pub(crate) enum Oper {
    Add(ORef),
    Sub(ORef),
    Mul(ORef),
    /// Negate
    Neg,
    /// Caution should be employed if required to use for power of 0 or 1
    Pow(i32),
    Sin,
    Cos,
    Sum(Vec<ORef>),
    Square,
    Variable(Var),
    Parameter(Par),
    Float(f64),
}

/// An expression that can be dynamically constructed.
///
/// Operator overloading is implemented for this struct alongside `Var`, `Par`
/// and `f64`. Once the final expression has been constructed, it can be
/// converted into a `ExprDyn`, which locks down the final form of the
/// expression, and calculates some additional information about its "sparsity"
/// (which is expensive to do, some only done once).
///
/// An `Expr` shouldn't be directly constructed, instead it is produced by
/// applying a valid operator (`+`, `-`, `*`, `powi`, `sin`, `cos`) to either
/// an existing operator, or to a `Var` or `Par` value.
///
/// ```
/// use descent::expr::{Var, Par};
/// use descent::expr::dynam::NumOps; // required to bring in powi usage
/// let x = Var(0);
/// let y = Var(1);
/// let p = Par(0);
///
/// let e = p * x - (5.0 * y.powi(2) + 4.0); // this produces an `Expr`
/// ```
///
/// One exception to this is for convenience, e.g., summing a bunch of terms:
///
/// ```
/// use descent::expr::Var;
/// use descent::expr::dynam::Expr;
/// let xs: Vec<Var> = (0..5).into_iter().map(|i| Var(i)).collect();
///
/// let mut e = Expr::from(0.0);
/// for &x in &xs {
///     e = e + x;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Expr {
    ops: Vec<Oper>,
}

impl From<Expr> for Expression {
    fn from(expr: Expr) -> Self {
        if let Some(Oper::Sum(ref js)) = expr.ops.last() {
            if js.len() > 4 {
                Expression::ExprDynSum(
                    expr.separate_sum()
                        .expect("Misformed expression cannot be separated")
                        .into_iter()
                        .map(|e| e.into())
                        .collect(),
                )
            } else {
                Expression::ExprDyn(expr.into())
            }
        } else {
            Expression::ExprDyn(expr.into())
        }
    }
}

// The following conventions are used for function argument names:
//
// - `v1` variable for first derivative
// - `v2` variables in second derivative
// - `vl2` variables for second derivative (list form)
// - `n` the value of a node
// - `nd` a first derivative for node
// - `na1` the "first" adjoint for node
// - `na2` a "second" adjoint for node
//
// These are postfixed with an `s` when a slice/vec of such values are
// supplied.
//
// In the mathematical working `n` represents a node, and it has the set of
// operands js.
impl Expr {
    /// Value of the expression.
    pub fn eval<R>(&self, ret: &R, ns: &mut Vec<f64>) -> f64
    where
        R: Retrieve,
    {
        use self::Oper::*;
        ns.resize(self.len(), 0.0);
        // Get values
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = ns.split_at_mut(i);
            let cur = &mut right[0]; // the i value from original
            match *op {
                Add(j) => *cur = left[i - 1] + left[i - j],
                Sub(j) => *cur = left[i - j] - left[i - 1],
                Mul(j) => *cur = left[i - 1] * left[i - j],
                Neg => *cur = -left[i - 1],
                Pow(pow) => *cur = left[i - 1].powi(pow),
                Sin => *cur = left[i - 1].sin(),
                Cos => *cur = left[i - 1].cos(),
                Sum(ref js) => {
                    *cur = left[i - 1];
                    for &j in js {
                        *cur += left[i - j];
                    }
                }
                Square => *cur = left[i - 1] * left[i - 1],
                Variable(v) => *cur = ret.var(v),
                Parameter(p) => *cur = ret.par(p),
                Float(val) => *cur = val,
            }
        }
        ns[self.len() - 1]
    }

    /// First derivative using forward method.
    ///
    /// `ns` must be the length of the expression.
    ///
    /// Could be made faster using loop unrolling if we passed in multiple
    /// variables to do at the same time.
    ///
    /// For a first derivative wrt `x_a`:
    ///
    /// ```text
    /// dn/dx_a = \sum_{j in js} dn/dn_j dn_j/dx_a
    /// ```
    ///
    /// The node derivatives `nds` are calculated from terminals to root. One
    /// pass is required for each variable:
    ///
    /// ```text
    /// nd = \sum_{j in js} nd_j dn/dn_j
    /// ```
    #[cfg(test)]
    fn der1_fwd(&self, v1: ID, ns: &[f64], nds: &mut Vec<f64>) -> f64 {
        use self::Oper::*;
        nds.resize(self.len(), 0.0);
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = nds.split_at_mut(i);
            let cur = &mut right[0]; // the i value from original
            match *op {
                Add(j) => *cur = left[i - 1] + left[i - j],
                Sub(j) => *cur = left[i - j] - left[i - 1],
                Mul(j) => *cur = left[i - 1] * ns[i - j] + left[i - j] * ns[i - 1],
                Neg => *cur = -left[i - 1],
                Pow(pow) => *cur = f64::from(pow) * left[i - 1] * ns[i - 1].powi(pow - 1),
                Sin => *cur = left[i - 1] * ns[i - 1].cos(),
                Cos => *cur = -left[i - 1] * ns[i - 1].sin(),
                Sum(ref js) => {
                    *cur = left[i - 1];
                    for &j in js {
                        *cur += left[i - j];
                    }
                }
                Square => *cur = 2.0 * left[i - 1] * ns[i - 1],
                Variable(Var(id)) => *cur = if id == v1 { 1.0 } else { 0.0 },
                _ => *cur = 0.0,
            }
        }
        nds[self.len() - 1]
    }

    /// First derivative using reverse method.
    ///
    /// `ns` must be the length of the expression.
    ///
    /// Assume each operator has only one dependent. If not then would need to
    /// rework.  Probably not an issue as can maybe just `na1_j +=` (would need
    /// to make sure they are set to 0 at start).
    ///
    /// ```text
    /// dn/dx_a = \sum_{j in js} dn/dn_j dn_j/dx_a
    /// ```
    ///
    /// The adjoints `na1s` are are calculated from the root to terminals. Once
    /// a variable terminal is reached, the ajoint value is combined values
    /// from the variable elsewhere in the expression.
    ///
    /// ```text
    /// na1_j = na1 dn/dn_j
    /// ```
    fn der1_rev(
        &self,
        v1s: &[ID],
        ns: &[f64],
        na1s: &mut Vec<f64>,
        ids: &mut HashMap<ID, f64>,
    ) -> Vec<f64> {
        use self::Oper::*;

        // Probably there is a faster way than this.
        ids.clear();
        for &id in v1s {
            ids.insert(id, 0.0);
        }

        // Go through in reverse
        na1s.resize(self.len(), 0.0);
        na1s[self.len() - 1] = 1.0;
        for (i, op) in self.ops.iter().enumerate().rev() {
            let (left, right) = na1s.split_at_mut(i);
            let cur = right[0]; // the i value from original
            match *op {
                Add(j) => {
                    left[i - 1] = cur;
                    left[i - j] = cur;
                }
                Sub(j) => {
                    // Take note of order where oth - pre
                    left[i - 1] = -cur;
                    left[i - j] = cur;
                }
                Mul(j) => {
                    left[i - 1] = ns[i - j] * cur;
                    left[i - j] = ns[i - 1] * cur;
                }
                Neg => {
                    left[i - 1] = -cur;
                }
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    left[i - 1] = f64::from(pow) * ns[i - 1].powi(pow - 1) * cur;
                }
                Sin => {
                    left[i - 1] = ns[i - 1].cos() * cur;
                }
                Cos => {
                    left[i - 1] = -ns[i - 1].sin() * cur;
                }
                Sum(ref js) => {
                    left[i - 1] = cur;
                    for &j in js {
                        left[i - j] = cur;
                    }
                }
                Square => {
                    left[i - 1] = 2.0 * ns[i - 1] * cur;
                }
                Variable(Var(id)) => {
                    if let Some(v) = ids.get_mut(&id) {
                        *v += cur;
                    }
                }
                _ => {}
            }
        }
        let mut der1 = Vec::new();
        for id in v1s {
            der1.push(*ids.get(id).unwrap());
        }
        der1
    }

    /// The second derivatives for given first derivative.
    ///
    /// `ns` and `nds` must be the length of the expression.
    ///
    /// ```text
    /// dn/dx_adx_a = \sum_{j \in js} d^2n_j/dx_adx_a
    ///     + \sum_{j, k \in js} d^2n/dn_jdn_k dn_j/dx_a dn_k/dx_b
    /// ```
    ///
    /// If we have precomputed the node derivatives wrt `x_b`: `nds`, then when
    /// we follow one path, we only need to pass on two "adjoint" values to
    /// each operand.  Given adjoints (na1, na2) for n, then the adjoint of
    /// operand j is:
    ///
    /// ```text
    /// na1_j = na1 dn/dn_j
    /// na2_j = na2 dn/dn_j + na1 s_j
    ///
    /// where s_j = \sum_{k \in js} d^2n/dn_jdn_k nd_k
    /// ```
    ///
    /// Once a terminal variable is reached, the second adjoint is combined
    /// with the second adjoint of the variable if it appears anywhere else in
    /// the expression
    #[cfg(test)]
    fn der2_rev(
        &self,
        dl2: &[ID],
        ns: &[f64],
        nds: &[f64],
        na1s: &mut Vec<f64>,
        na2s: &mut Vec<f64>,
        ids: &mut HashMap<ID, f64>,
    ) -> Vec<f64> {
        use self::Oper::*;

        // Probably there is a faster way than this.
        ids.clear();
        for &id in dl2 {
            ids.insert(id, 0.0);
        }

        // Go through in reverse
        na1s.resize(self.len(), 0.0);
        na1s[self.len() - 1] = 1.0;
        na2s.resize(self.len(), 0.0);
        na2s[self.len() - 1] = 0.0;
        for (i, op) in self.ops.iter().enumerate().rev() {
            let (l1, r1) = na1s.split_at_mut(i);
            let c1 = r1[0];
            let (l2, r2) = na2s.split_at_mut(i);
            let c2 = r2[0];
            match *op {
                Add(j) => {
                    l1[i - 1] = c1;
                    l2[i - 1] = c2;
                    l1[i - j] = c1;
                    l2[i - j] = c2;
                }
                Sub(j) => {
                    // Take note of order where oth - pre
                    l1[i - 1] = -c1;
                    l2[i - 1] = -c2;
                    l1[i - j] = c1;
                    l2[i - j] = c2;
                }
                Mul(j) => {
                    l1[i - 1] = c1 * ns[i - j];
                    l2[i - 1] = c2 * ns[i - j] + c1 * nds[i - j];
                    l1[i - j] = c1 * ns[i - 1];
                    l2[i - j] = c2 * ns[i - 1] + c1 * nds[i - 1];
                }
                Neg => {
                    l1[i - 1] = -c1;
                    l2[i - 1] = -c2;
                }
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    let vald = f64::from(pow) * ns[i - 1].powi(pow - 1);
                    let valdd = f64::from(pow * (pow - 1)) * ns[i - 1].powi(pow - 2);
                    l1[i - 1] = c1 * vald;
                    l2[i - 1] = c2 * vald + c1 * valdd * nds[i - 1];
                }
                Sin => {
                    l1[i - 1] = c1 * ns[i - 1].cos();
                    l2[i - 1] = c2 * ns[i - 1].cos() - c1 * ns[i - 1].sin() * nds[i - 1];
                }
                Cos => {
                    l1[i - 1] = -c1 * ns[i - 1].sin();
                    l2[i - 1] = -c2 * ns[i - 1].sin() - c1 * ns[i - 1].cos() * nds[i - 1];
                }
                Sum(ref js) => {
                    l1[i - 1] = c1;
                    l2[i - 1] = c2;
                    for &j in js {
                        l1[i - j] = c1;
                        l2[i - j] = c2;
                    }
                }
                Square => {
                    l1[i - 1] = c1 * 2.0 * ns[i - 1];
                    l2[i - 1] = c2 * 2.0 * ns[i - 1] + c1 * 2.0 * nds[i - 1];
                }
                Variable(Var(id)) => {
                    if let Some(v) = ids.get_mut(&id) {
                        *v += c2;
                    }
                }
                _ => {}
            }
        }
        let mut der2 = Vec::new();
        for id in dl2 {
            der2.push(*ids.get(id).unwrap());
        }
        der2
    }

    /// Value, and derivatives using forward method.
    fn full_fwd<'a, R>(
        &self,
        v1s: &[ID],
        v2s: &[(usize, usize)],
        ret: &R,
        cols: &'a mut Vec<Column>,
    ) -> &'a Column
    where
        R: Retrieve,
    {
        use self::Oper::*;
        // Only resize up
        if cols.len() < self.len() {
            cols.resize(self.len(), Column::new());
        }
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = cols.split_at_mut(i);
            let cur = &mut right[0]; // the i value from original
            cur.der1.resize(v1s.len(), 0.0);
            cur.der2.resize(v2s.len(), 0.0);
            match *op {
                Add(j) => {
                    let pre = &left[i - 1];
                    let oth = &left[i - j];
                    cur.val = pre.val + oth.val;
                    for ((c, p), o) in cur
                        .der1
                        .iter_mut()
                        .zip(pre.der1.iter())
                        .zip(oth.der1.iter())
                    {
                        *c = p + o;
                    }
                    for ((c, p), o) in cur
                        .der2
                        .iter_mut()
                        .zip(pre.der2.iter())
                        .zip(oth.der2.iter())
                    {
                        *c = p + o;
                    }
                }
                Sub(j) => {
                    // Take note of order where oth - pre
                    let pre = &left[i - 1];
                    let oth = &left[i - j];
                    cur.val = oth.val - pre.val;
                    for ((c, p), o) in cur
                        .der1
                        .iter_mut()
                        .zip(pre.der1.iter())
                        .zip(oth.der1.iter())
                    {
                        *c = o - p;
                    }
                    for ((c, p), o) in cur
                        .der2
                        .iter_mut()
                        .zip(pre.der2.iter())
                        .zip(oth.der2.iter())
                    {
                        *c = o - p;
                    }
                }
                Mul(j) => {
                    let pre = &left[i - 1];
                    let oth = &left[i - j];
                    cur.val = pre.val * oth.val;
                    for k in 0..(v1s.len()) {
                        cur.der1[k] = pre.der1[k] * oth.val + pre.val * oth.der1[k];
                    }
                    for (((c, p), o), &(k1, k2)) in cur
                        .der2
                        .iter_mut()
                        .zip(pre.der2.iter())
                        .zip(oth.der2.iter())
                        .zip(v2s.iter())
                    {
                        *c = p * oth.val
                            + pre.val * o
                            + pre.der1[k1] * oth.der1[k2]
                            + pre.der1[k2] * oth.der1[k1];
                    }
                }
                Neg => {
                    let pre = &left[i - 1];
                    cur.val = -pre.val;
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = -p;
                    }
                    for (c, p) in cur.der2.iter_mut().zip(pre.der2.iter()) {
                        *c = -p;
                    }
                }
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    let pre = &left[i - 1];
                    cur.val = pre.val.powi(pow);
                    let vald = f64::from(pow) * pre.val.powi(pow - 1);
                    let valdd = f64::from(pow * (pow - 1)) * pre.val.powi(pow - 2);
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = p * vald;
                    }
                    for ((c, p), &(k1, k2)) in
                        cur.der2.iter_mut().zip(pre.der2.iter()).zip(v2s.iter())
                    {
                        *c = p * vald + pre.der1[k1] * pre.der1[k2] * valdd;
                    }
                }
                Sin => {
                    let pre = &left[i - 1];
                    cur.val = pre.val.sin();
                    let valcos = pre.val.cos();
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = p * valcos;
                    }
                    for ((c, p), &(k1, k2)) in
                        cur.der2.iter_mut().zip(pre.der2.iter()).zip(v2s.iter())
                    {
                        *c = p * valcos - pre.der1[k1] * pre.der1[k2] * cur.val;
                    }
                }
                Cos => {
                    let pre = &left[i - 1];
                    cur.val = pre.val.cos();
                    let valsin = pre.val.sin();
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = -p * valsin;
                    }
                    for ((c, p), &(k1, k2)) in
                        cur.der2.iter_mut().zip(pre.der2.iter()).zip(v2s.iter())
                    {
                        *c = -p * valsin - pre.der1[k1] * pre.der1[k2] * cur.val;
                    }
                }
                Sum(ref js) => {
                    let pre = &left[i - 1];
                    cur.val = pre.val;
                    for &j in js {
                        let oth = &left[i - j];
                        cur.val += oth.val;
                    }

                    // Could just do a copy?
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = *p;
                    }
                    for &j in js {
                        let oth = &left[i - j];
                        for (c, o) in cur.der1.iter_mut().zip(oth.der1.iter()) {
                            *c += *o;
                        }
                    }
                    // Could just do a copy?
                    for (c, p) in cur.der2.iter_mut().zip(pre.der2.iter()) {
                        *c = *p;
                    }
                    for &j in js {
                        let oth = &left[i - j];
                        for (c, o) in cur.der2.iter_mut().zip(oth.der2.iter()) {
                            *c += *o;
                        }
                    }
                    //for k in 0..(v1s.len()) {
                    //    cur.der1[k] = pre.der1[k];
                    //    for &j in js {
                    //        let oth = &left[i - j];
                    //        cur.der1[k] += oth.der1[k];
                    //    }
                    //}
                    //for k in 0..(v2s.len()) {
                    //    cur.der2[k] = pre.der2[k];
                    //    for &j in js {
                    //        let oth = &left[i - j];
                    //        cur.der2[k] += oth.der2[k];
                    //    }
                    //}
                }
                Square => {
                    let pre = &left[i - 1];
                    cur.val = pre.val * pre.val;
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = 2.0 * p * pre.val;
                    }
                    for ((c, p), &(k1, k2)) in
                        cur.der2.iter_mut().zip(pre.der2.iter()).zip(v2s.iter())
                    {
                        *c = 2.0 * p * pre.val + 2.0 * pre.der1[k1] * pre.der1[k2];
                    }
                }
                Variable(Var(id)) => {
                    cur.val = ret.var(Var(id));
                    for (c, did) in cur.der1.iter_mut().zip(v1s.iter()) {
                        *c = if id == *did { 1.0 } else { 0.0 };
                    }
                    for c in &mut cur.der2 {
                        *c = 0.0;
                    }
                }
                Parameter(Par(id)) => {
                    cur.val = ret.par(Par(id));
                    for c in &mut cur.der1 {
                        *c = 0.0;
                    }
                    for c in &mut cur.der2 {
                        *c = 0.0;
                    }
                }
                Float(val) => {
                    cur.val = val;
                    for c in &mut cur.der1 {
                        *c = 0.0;
                    }
                    for c in &mut cur.der2 {
                        *c = 0.0;
                    }
                }
            }
        }
        &cols[self.len() - 1]
    }

    /// Value, and derivatives using both forward and reverse method.
    ///
    /// `v1s` and `vl2s` must be same length.
    #[cfg(test)]
    fn full_fwd_rev<R>(&self, v1s: &[ID], vl2s: &[Vec<ID>], ret: &R, ws: &mut WorkSpace) -> Column
    where
        R: Retrieve,
    {
        let mut col = Column::new();

        col.val = self.eval(ret, &mut ws.ns);

        for (&id, oids) in v1s.iter().zip(vl2s.iter()) {
            if !oids.is_empty() {
                col.der1.push(self.der1_fwd(id, &ws.ns, &mut ws.nds));
                col.der2.append(&mut self.der2_rev(
                    oids,
                    &ws.ns,
                    &ws.nds,
                    &mut ws.na1s,
                    &mut ws.na2s,
                    &mut ws.ids,
                ));
            }
        }
        // Check if all first derivatives were calculated, and if not just use
        // the reverse method.
        if col.der1.len() < v1s.len() {
            col.der1 = self.der1_rev(v1s, &ws.ns, &mut ws.na1s, &mut ws.ids);
        }

        col
    }

    /// Get information about the expression.
    pub fn get_info(&self) -> ExprInfo {
        use self::Oper::*;
        let mut degs: Vec<Degree> = Vec::new();
        for (i, op) in self.ops.iter().enumerate() {
            let d = match *op {
                Add(j) | Sub(j) => degs[i - 1].union(&degs[i - j]),
                Mul(j) => degs[i - 1].cross(&degs[i - j]),
                Neg => degs[i - 1].clone(),
                Pow(p) => {
                    // Even though shouldn't have 0 or 1, might as well match
                    // anyway
                    match p {
                        0 => Degree::new(),
                        1 => degs[i - 1].clone(),
                        2 => degs[i - 1].cross(&degs[i - 1]),
                        _ => degs[i - 1].highest(),
                    }
                }
                Sum(ref js) => {
                    let mut deg = degs[i - 1].clone();
                    for &j in js {
                        deg = deg.union(&degs[i - j]);
                    }
                    deg
                }
                Square => degs[i - 1].cross(&degs[i - 1]),
                Sin | Cos => degs[i - 1].highest(),
                Variable(Var(id)) => Degree::with_id(id),
                _ => Degree::new(),
            };
            degs.push(d);
        }

        match degs.pop() {
            Some(d) => d.into(),
            None => ExprInfo::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push operation onto expression.
    fn add_op(&mut self, op: Oper) {
        self.ops.push(op);
    }

    /// Append another expression.
    fn append(&mut self, mut other: Expr) {
        self.ops.append(&mut other.ops);
    }

    /// Must not be empty.
    fn extract_add_orefs(&mut self, js: &mut Vec<ORef>) {
        match self.ops.pop().unwrap() {
            Oper::Add(j) => {
                js.push(j);
            }
            Oper::Sub(j) => {
                self.ops.push(Oper::Neg);
                js.push(j + 1);
            }
            Oper::Sum(mut ljs) => {
                js.append(&mut ljs);
            }
            op => {
                self.ops.push(op); // push back on
            }
        }
    }

    /// Separate sum into vector of expressions.
    fn separate_sum(self) -> Option<Vec<Expr>> {
        let mut ops = self.ops;
        if let Some(Oper::Sum(oref)) = ops.pop() {
            let mut v = Vec::new();
            while !ops.is_empty() {
                let i = find_term_start(&ops);
                v.push(Expr {
                    ops: ops.drain(i..).collect(),
                });
            }
            if oref.len() + 1 != v.len() {
                panic!("Unexpected separation of sum operation");
            }
            Some(v)
        } else {
            None
        }
    }
}

impl ExprDyn {
    /// Constant derivatives by method auto selection.
    ///
    /// It is the users resposibility to pass in a valid `Retrieve` for the
    /// variables and parameters present in the expression.
    pub fn auto_const<R>(&self, store: &R, ws: &mut WorkSpace) -> Column
    where
        R: Retrieve,
    {
        let expr = &self.expr;
        let info = &self.info;
        let mut col = Column::new();
        if !info.lin.is_empty() {
            expr.eval(store, &mut ws.ns);
            col.der1 = expr.der1_rev(&info.lin, &ws.ns, &mut ws.na1s, &mut ws.ids);
        }
        if !info.quad.is_empty() {
            col.der2 = expr
                .full_fwd(&info.nlin, &info.quad, store, &mut ws.cols)
                .der2
                .clone();
            // If using this need to make sure expr has had set_lists() called
            //col.der2 = self.full_fwd_rev(&info.nlin, &info.quad_list, store,
            //                             ws).der2;
        }
        col
    }

    /// Dynamic values/derivatives by method auto selection.
    ///
    /// It is the users resposibility to pass in a valid `Retrieve` for the
    /// variables and parameters present in the expression.
    pub fn auto_dynam<R>(&self, store: &R, ws: &mut WorkSpace) -> Column
    where
        R: Retrieve,
    {
        let expr = &self.expr;
        let info = &self.info;
        if info.nlin.is_empty() {
            let mut col = Column::new();
            col.val = expr.eval(store, &mut ws.ns);
            col
        } else if info.nquad.is_empty() {
            let mut col = Column::new();
            col.val = expr.eval(store, &mut ws.ns);
            col.der1 = expr.der1_rev(&info.nlin, &ws.ns, &mut ws.na1s, &mut ws.ids);
            col
        } else {
            expr.full_fwd(&info.nlin, &info.nquad, store, &mut ws.cols)
                .clone()
            // If using this need to make sure expr has had set_lists() called
            //self.full_fwd_rev(&info.nlin, &info.nquad_list, store, ws);
        }
    }

    //auto_sep_dynam(&self, ginfo: &ExprInfo, sinfos: &Vec<SegInfo>,
    //               store: &R, ws: &mut WorkSpace) -> Column {
    //    let mut gcol: Column::new();
    //    // Should call reverse method to get first derivatives and value
    //    //gcol.der1.resize(ginfo.nlin.len(), 0.0);
    //    gcol.der2.resize(ginfo.nquad.len(), 0.0);
    //    for sinfo in &sinfos {
    //        let col = self.full_fwd(&sinfo.info.nlin, &sinfo.info.nquad,
    //                                store, &mut ws.cols);
    //        for (i, d) in sinfo.nquad_map.iter().zip(col.der2.iter()) {
    //            gcol.der2[i] += d;
    //        }
    //    }
    //}
}

/// Cross two sets of IDs.
fn cross_ids(id1s: &HashSet<ID>, id2s: &HashSet<ID>, target: &mut HashSet<(ID, ID)>) {
    for &id1 in id1s {
        for &id2 in id2s {
            target.insert(super::order(id1, id2));
        }
    }
}

/// Representation of the degree of variables from an expression.
///
/// - `lin` and `nlin` must be disjoint
/// - `quad` and `nquad` must be disjoint
/// - IDs in pairs must be ordered
/// - all IDs in `quad` and `nquad` must be in `nlin`
#[derive(Debug, PartialEq, Clone, Default)]
struct Degree {
    lin: HashSet<ID>,
    nlin: HashSet<ID>,
    quad: HashSet<(ID, ID)>,
    nquad: HashSet<(ID, ID)>,
}

impl Degree {
    fn new() -> Self {
        Self::default()
    }

    fn with_id(id: ID) -> Degree {
        let mut d = Degree::default();
        d.lin.insert(id);
        d
    }

    fn is_empty(&self) -> bool {
        self.lin.is_empty() && self.quad.is_empty() && self.nquad.is_empty()
    }

    fn union(&self, other: &Degree) -> Degree {
        let mut deg = Degree::new();
        deg.lin = self.lin.union(&(other.lin)).cloned().collect();
        deg.nlin = self.nlin.union(&(other.nlin)).cloned().collect();
        deg.quad = self.quad.union(&(other.quad)).cloned().collect();
        deg.nquad = self.nquad.union(&(other.nquad)).cloned().collect();

        deg.lin = deg.lin.difference(&(deg.nlin)).cloned().collect();
        deg.quad = deg.quad.difference(&(deg.nquad)).cloned().collect();
        deg
    }

    /// Cross all elements in degree.
    fn cross(&self, other: &Degree) -> Degree {
        if self.is_empty() {
            other.clone()
        } else if other.is_empty() {
            self.clone()
        } else {
            // If here, both sides have at least one entry.
            // Therefore all promoted one level.
            let mut deg = Degree::new();

            // all nlin move over
            for &id in &self.nlin {
                deg.nlin.insert(id);
            }

            for &id in &other.nlin {
                deg.nlin.insert(id);
            }

            // lin will empty into nlin
            for &id in &self.lin {
                deg.nlin.insert(id);
            }

            for &id in &other.lin {
                deg.nlin.insert(id);
            }

            // quad and nquad will transfer over to nquad
            for &p in &self.nquad {
                deg.nquad.insert(p);
            }

            for &p in &other.nquad {
                deg.nquad.insert(p);
            }

            for &p in &self.quad {
                deg.nquad.insert(p);
            }

            for &p in &other.quad {
                deg.nquad.insert(p);
            }

            // contract ensures new quad values do not appear in nquad
            cross_ids(&(self.lin), &(other.lin), &mut (deg.quad));
            cross_ids(&(self.lin), &(other.nlin), &mut (deg.nquad));
            cross_ids(&(self.nlin), &(other.lin), &mut (deg.nquad));
            cross_ids(&(self.nlin), &(other.nlin), &mut (deg.nquad));

            deg
        }
    }

    /// Promote all combinations to highest degree.
    fn highest(&self) -> Degree {
        let mut deg = Degree::new();

        deg.nlin = self.lin.union(&(self.nlin)).cloned().collect();

        // Could same time here due to symmetry...
        cross_ids(&(deg.nlin), &(deg.nlin), &mut (deg.nquad));

        deg
    }
}

impl From<Degree> for ExprInfo {
    fn from(d: Degree) -> ExprInfo {
        let mut info = ExprInfo::new();
        info.lin = d.lin.into_iter().collect();
        info.nlin = d.nlin.into_iter().collect();
        info.lin.sort();
        info.nlin.sort();

        let mut id_to_ind: HashMap<ID, usize> = HashMap::new();
        for (i, &id) in info.nlin.iter().enumerate() {
            id_to_ind.insert(id, i);
        }

        // Converting from variable IDs to local indices into nlin
        for (id1, id2) in d.quad {
            info.quad.push((id_to_ind[&id1], id_to_ind[&id2]));
        }

        for (id1, id2) in d.nquad {
            info.nquad.push((id_to_ind[&id1], id_to_ind[&id2]));
        }

        info.quad.sort();
        info.nquad.sort();

        // Not always required...
        //info.set_lists();

        info
    }
}

/// Find left-most index of a separable term at end of slice of operands.
fn find_term_start(ops: &[Oper]) -> usize {
    if ops.is_empty() {
        0
    } else {
        let i: usize = ops.len() - 1;
        match ops[i] {
            Oper::Add(j) | Oper::Sub(j) | Oper::Mul(j) => std::cmp::min(
                find_term_start(&ops[..i]),
                find_term_start(&ops[..i + 1 - j]),
            ),
            Oper::Neg | Oper::Pow(_) | Oper::Sin | Oper::Cos | Oper::Square => {
                find_term_start(&ops[..i])
            }
            Oper::Variable(_) | Oper::Parameter(_) | Oper::Float(_) => i,
            Oper::Sum(ref js) => {
                let mut k = find_term_start(&ops[..i]);
                for &j in js {
                    k = std::cmp::min(k, find_term_start(&ops[..i + 1 - j]));
                }
                k
            }
        }
    }
}

impl From<Var> for Expr {
    fn from(v: Var) -> Expr {
        Expr {
            ops: vec![self::Oper::Variable(v)],
        }
    }
}

impl From<Par> for Expr {
    fn from(p: Par) -> Expr {
        Expr {
            ops: vec![self::Oper::Parameter(p)],
        }
    }
}

impl From<f64> for Expr {
    fn from(v: f64) -> Expr {
        Expr {
            ops: vec![self::Oper::Float(v)],
        }
    }
}

impl From<i32> for Expr {
    fn from(v: i32) -> Expr {
        Expr {
            ops: vec![self::Oper::Float(f64::from(v))],
        }
    }
}

// Have to use trait because straight fn overloading not possible
/// Trait for using numeric operators such as `.powi(i32)`, `.cos()`, `.sin()`
/// in dynamic expressions.
pub trait NumOps {
    fn powi(self, p: i32) -> Expr;
    fn sin(self) -> Expr;
    fn cos(self) -> Expr;
}

impl NumOps for Expr {
    fn powi(mut self, p: i32) -> Expr {
        // When empty don't do anything
        if self.ops.is_empty() {
            return self;
        }

        // Match now so don't have to later
        match p {
            0 => self.add_op(Oper::Float(1.0)),
            1 => (), //don't add anything
            2 => self.add_op(Oper::Square),
            _ => self.add_op(Oper::Pow(p)),
        }
        self
    }

    fn sin(mut self) -> Expr {
        // When empty don't do anything
        if !self.ops.is_empty() {
            self.add_op(Oper::Sin);
        }
        self
    }

    fn cos(mut self) -> Expr {
        // When empty don't do anything
        if !self.ops.is_empty() {
            self.add_op(Oper::Cos);
        }
        self
    }
}

impl NumOps for Var {
    fn powi(self, p: i32) -> Expr {
        Expr::from(self).powi(p)
    }

    fn sin(self) -> Expr {
        Expr::from(self).sin()
    }

    fn cos(self) -> Expr {
        Expr::from(self).cos()
    }
}

impl NumOps for Par {
    fn powi(self, p: i32) -> Expr {
        Expr::from(self).powi(p)
    }

    fn sin(self) -> Expr {
        Expr::from(self).sin()
    }

    fn cos(self) -> Expr {
        Expr::from(self).cos()
    }
}

macro_rules! binary_ops_to_expr {
    ( $T:ident, $f:ident, $U:ty, $V:ty ) => {
        impl $T<$V> for $U {
            type Output = Expr;

            fn $f(self, other: $V) -> Expr {
                Expr::from(self).$f(Expr::from(other))
            }
        }
    };
}

macro_rules! binary_ops_with_expr {
    ( $T:ident, $f:ident, $U:ty ) => {
        impl $T<Expr> for $U {
            type Output = Expr;

            fn $f(self, other: Expr) -> Expr {
                Expr::from(self).$f(other)
            }
        }

        impl $T<$U> for Expr {
            type Output = Expr;

            fn $f(self, other: $U) -> Expr {
                self.$f(Expr::from(other))
            }
        }

        impl<'a> $T<&'a Expr> for $U {
            type Output = Expr;

            fn $f(self, other: &'a Expr) -> Expr {
                Expr::from(self).$f(other.clone())
            }
        }

        impl<'a> $T<$U> for &'a Expr {
            type Output = Expr;

            fn $f(self, other: $U) -> Expr {
                self.clone().$f(Expr::from(other))
            }
        }
    };
}

binary_ops_to_expr!(Add, add, Var, f64);
binary_ops_to_expr!(Add, add, f64, Var);
binary_ops_to_expr!(Add, add, Par, f64);
binary_ops_to_expr!(Add, add, f64, Par);
binary_ops_to_expr!(Add, add, Par, Var);
binary_ops_to_expr!(Add, add, Var, Par);
binary_ops_to_expr!(Add, add, Var, Var);
binary_ops_to_expr!(Add, add, Par, Par);

binary_ops_to_expr!(Sub, sub, Var, f64);
binary_ops_to_expr!(Sub, sub, f64, Var);
binary_ops_to_expr!(Sub, sub, Par, f64);
binary_ops_to_expr!(Sub, sub, f64, Par);
binary_ops_to_expr!(Sub, sub, Par, Var);
binary_ops_to_expr!(Sub, sub, Var, Par);
binary_ops_to_expr!(Sub, sub, Var, Var);
binary_ops_to_expr!(Sub, sub, Par, Par);

binary_ops_to_expr!(Mul, mul, Var, f64);
binary_ops_to_expr!(Mul, mul, f64, Var);
binary_ops_to_expr!(Mul, mul, Par, f64);
binary_ops_to_expr!(Mul, mul, f64, Par);
binary_ops_to_expr!(Mul, mul, Par, Var);
binary_ops_to_expr!(Mul, mul, Var, Par);
binary_ops_to_expr!(Mul, mul, Var, Var);
binary_ops_to_expr!(Mul, mul, Par, Par);

binary_ops_with_expr!(Add, add, Var);
binary_ops_with_expr!(Add, add, Par);
binary_ops_with_expr!(Add, add, f64);
binary_ops_with_expr!(Sub, sub, Var);
binary_ops_with_expr!(Sub, sub, Par);
binary_ops_with_expr!(Sub, sub, f64);
binary_ops_with_expr!(Mul, mul, Var);
binary_ops_with_expr!(Mul, mul, Par);
binary_ops_with_expr!(Mul, mul, f64);

impl std::ops::Add<Expr> for Expr {
    type Output = Expr;

    fn add(mut self, mut other: Expr) -> Expr {
        // Assuming add on empty Expr is like add by 0.0
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            let mut js = Vec::new();
            other.extract_add_orefs(&mut js);
            let njs_other = js.len();
            let n = other.len();
            js.push(1 + n);
            self.extract_add_orefs(&mut js);
            for j in js.iter_mut().skip(njs_other + 1) {
                *j += n;
            }

            self.append(other);
            if js.len() > 1 {
                self.add_op(Oper::Sum(js));
            } else {
                self.add_op(Oper::Add(n + 1));
            }
            // Use this to turn off sum
            //let j = other.len() + 1;
            //self.append(other);
            //self.add_op(Oper::Add(j));
            self
        }
    }
}

impl std::ops::Sub<Expr> for Expr {
    type Output = Expr;

    fn sub(mut self, mut other: Expr) -> Expr {
        // Assuming sub on empty Expr is like add by 0.0
        if self.ops.is_empty() {
            other.add_op(Oper::Neg); // negate second argument
            other
        } else if other.ops.is_empty() {
            self
        } else {
            // Unlike Add cannot combine Add and Sum entries of other
            let mut js = Vec::new();
            let n = other.len();
            js.push(1 + n + 1); // Will be adding Neg
            self.extract_add_orefs(&mut js);
            for j in js.iter_mut().skip(1) {
                *j += n + 1;
            }

            self.append(other);
            if js.len() > 1 {
                self.add_op(Oper::Neg);
                self.add_op(Oper::Sum(js));
            } else {
                self.add_op(Oper::Sub(n + 1));
            }
            // Use this to turn off sum
            //let j = other.len() + 1;
            //self.append(other);
            //self.add_op(Oper::Sub(j));
            self
        }
    }
}

impl std::ops::Mul<Expr> for Expr {
    type Output = Expr;

    fn mul(mut self, other: Expr) -> Expr {
        // Assuming mul on empty Expr is like mul by 1.0
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            let j = other.len() + 1;
            self.append(other);
            self.add_op(Oper::Mul(j));
            self
        }
    }
}

/// Workspace used for performing expression AD.
///
/// Used to save on allocations with many repeated calls.
#[derive(Debug, Clone, Default)]
pub struct WorkSpace {
    pub cols: Vec<Column>,
    pub ns: Vec<f64>,
    pub nds: Vec<f64>,
    pub na1s: Vec<f64>,
    pub na2s: Vec<f64>,
    pub ids: HashMap<ID, f64>,
}

impl WorkSpace {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::super::Store;
    use super::*;

    #[test]
    fn operations() {
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        let mut e = Expr::from(5.0);
        e = 5.0 + e;
        e = Var(0) + e;
        e = e * Par(0);
        //println!("{:?}", e);

        let info = e.get_info();
        //println!("{:?}", info);

        let mut ws = WorkSpace::new();
        let col = e.full_fwd(&info.lin, &Vec::new(), &store, &mut ws.cols);

        assert_eq!(col.val, 60.0);
        assert_eq!(col.der1[0], 4.0);
    }

    #[test]
    fn variety() {
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let e = Var(0) + (1.0 - Var(1)).powi(2);
        let info = e.get_info();
        let mut ws = WorkSpace::new();

        // Get constant first derivatives
        // Call on parameter change
        // Copy out and store first derivatives
        {
            let col = e.full_fwd(&info.lin, &Vec::new(), &store, &mut ws.cols);
            assert_eq!(col.val, 14.0);
            assert_eq!(col.der1[0], 1.0); // Var(0)
        }

        // Get constant second derivatives
        // Call on parameter change
        // Copy out and store second derivatives
        {
            let col = e.full_fwd(&info.nlin, &info.quad, &store, &mut ws.cols);
            assert_eq!(col.val, 14.0);
            assert_eq!(col.der1[0], 6.0); // Var(1)
            assert_eq!(col.der2[0], 2.0); // Var(1), Var(1)
        }

        // Get dynamic derivatives
        // Call every time
        {
            let col = e.full_fwd(&info.nlin, &info.nquad, &store, &mut ws.cols);
            assert_eq!(col.val, 14.0);
            assert_eq!(col.der1[0], 6.0); // Var(1)
        }
    }

    #[test]
    fn degree() {
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let e = Var(0) + (1.0 - Var(1)).powi(2);
        let mut info = e.get_info();
        info.set_lists();

        assert_eq!(
            info,
            ExprInfo {
                lin: vec![0],
                nlin: vec![1],
                quad: vec![(0, 0)],
                nquad: vec![],
                quad_list: vec![vec![1]],
                nquad_list: vec![vec![]],
            }
        );
    }

    #[test]
    fn sum() {
        let mut store = Store::new();
        store.vars.push(5.0);

        let e = 5.0 + Var(0) + Var(0) + Var(0);
        let info = e.get_info();

        //println!("{:?}", e);
        //println!("{:?}", info);
        let mut ws = WorkSpace::new();
        let col = e.full_fwd(&info.lin, &Vec::new(), &store, &mut ws.cols);

        assert_eq!(e.len(), 5);
        assert_eq!(col.val, 20.0);
    }

    #[test]
    fn sum_sub() {
        let mut store = Store::new();
        store.vars.push(5.0);

        let e = 5.0 - Var(0) - Var(0) - Var(0);
        let info = e.get_info();

        //println!("{:?}", e);
        //println!("{:?}", info);
        let mut ws = WorkSpace::new();
        let col = e.full_fwd(&info.lin, &Vec::new(), &store, &mut ws.cols);

        assert_eq!(e.len(), 8);
        assert_eq!(col.val, -10.0);
    }

    #[test]
    fn sin() {
        let mut store = Store::new();
        store.vars.push(5.0);

        let mut ws = WorkSpace::new();

        let e = (2.0 * Var(0)).sin();
        let info = e.get_info();
        //println!("{:?}", f);
        //println!("{:?}", info);
        assert_eq!(info.nlin.len(), 1);
        assert_eq!(info.nquad.len(), 1);

        let col = e.full_fwd(&info.nlin, &info.nquad, &store, &mut ws.cols);

        assert_eq!(col.val, 10.0_f64.sin());
        assert_eq!(col.der1[0], 2.0 * (10.0_f64.cos()));
        assert_eq!(col.der2[0], -4.0 * (10.0_f64.sin()));
    }

    #[test]
    fn cos() {
        let mut store = Store::new();
        store.vars.push(5.0);

        let mut ws = WorkSpace::new();

        let e = (2.0 * Var(0)).cos();
        let info = e.get_info();
        //println!("{:?}", f);
        //println!("{:?}", info);
        assert_eq!(info.nlin.len(), 1);
        assert_eq!(info.nquad.len(), 1);

        let col = e.full_fwd(&info.nlin, &info.nquad, &store, &mut ws.cols);

        assert_eq!(col.val, 10.0_f64.cos());
        assert_eq!(col.der1[0], -2.0 * (10.0_f64.sin()));
        assert_eq!(col.der2[0], -4.0 * (10.0_f64.cos()));
    }

    #[test]
    fn reverse() {
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let x1 = Var(0);
        let x2 = Var(1);

        let e = x1 * x2 + x1.sin();
        let mut ws = WorkSpace::new();
        //let info = e.get_info();
        //println!("{:?}", e);
        //println!("{:?}", info);
        let v = e.eval(&store, &mut ws.ns);
        let der1 = e.der1_rev(&vec![0, 1], &ws.ns, &mut ws.na1s, &mut ws.ids);

        assert_eq!(v, 20.0 + 5.0_f64.sin());
        assert_eq!(der1[0], 5.0_f64.cos() + 4.0);
        assert_eq!(der1[1], 5.0);
    }

    #[test]
    fn forward_reverse() {
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let x1 = Var(0);
        let x2 = Var(1);

        let e = x1 * x2 + x1.sin();
        let mut ws = WorkSpace::new();
        let mut info = e.get_info();
        info.set_lists();
        //println!("{:?}", e);
        //println!("{:?}", info);
        e.eval(&store, &mut ws.ns);
        let quad_col = e.full_fwd_rev(&info.nlin, &info.quad_list, &store, &mut ws);
        let nquad_col = e.full_fwd_rev(&info.nlin, &info.nquad_list, &store, &mut ws);

        assert_eq!(quad_col.val, 20.0 + 5.0_f64.sin());
        assert_eq!(quad_col.der1.len(), 2);
        assert_eq!(quad_col.der1[0], 5.0_f64.cos() + 4.0);
        assert_eq!(quad_col.der1[1], 5.0);
        assert_eq!(quad_col.der2.len(), 1); // quad
        assert_eq!(quad_col.der2[0], 1.0);
        assert_eq!(nquad_col.val, 20.0 + 5.0_f64.sin());
        assert_eq!(nquad_col.der1.len(), 2);
        assert_eq!(nquad_col.der1[0], 5.0_f64.cos() + 4.0);
        assert_eq!(nquad_col.der1[1], 5.0);
        assert_eq!(nquad_col.der2.len(), 1); // quad
        assert_eq!(nquad_col.der2[0], -5.0_f64.sin());
    }

    #[test]
    fn separation() {
        let vars: Vec<_> = (0..6).into_iter().map(|i| Var(i)).collect();
        let mut store = Store::new();
        for _ in &vars {
            store.vars.push(3.0);
        }
        let mut ws = WorkSpace::new();

        let mut obj = (vars[0] - 1.0).powi(2);
        for v in vars.iter().skip(1) {
            obj = obj + (*v - 1.0).powi(2);
        }
        let e: Expression = obj.into();
        if let Expression::ExprDynSum(es) = e {
            for e in es {
                assert!(e.info.nlin.len() == 1);
                assert!(e.expr.eval(&store, &mut ws.ns) == 4.0);
            }
        } else {
            panic!("Didn't get a ExprDynSum when expected");
        }
    }

    #[bench]
    fn auto_const(b: &mut test::Bencher) {
        let n = 50;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Var(i));
            store.vars.push(0.5);
        }
        let mut e = Expr::from(0.0);
        for &x in &xs {
            e = e + 3.0 * (x - 1.0).powi(2) + 5.0;
        }
        let mut ws = WorkSpace::new();
        //println!("{:?}", e);
        let e: ExprDyn = e.into();
        b.iter(|| {
            e.auto_const(&store, &mut ws);
        });
    }

    #[bench]
    fn auto_dynam(b: &mut test::Bencher) {
        let n = 50;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Var(i));
            store.vars.push(0.5);
        }
        let mut e = Expr::from(0.0);
        for &x in &xs {
            e = e + 3.0 * (x - 1.0).powi(2) + 5.0;
        }
        let mut ws = WorkSpace::new();
        //println!("{:?}", e);
        let e: ExprDyn = e.into();
        b.iter(|| {
            e.auto_dynam(&store, &mut ws);
        });
    }
}
