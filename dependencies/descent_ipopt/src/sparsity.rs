// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use descent::expr::Expression;
use descent::expr::{Var, ID};
use fnv::FnvHashMap;

pub type JacSparsity = FnvHashMap<(usize, ID), usize>;
pub type HesSparsity = FnvHashMap<(ID, ID), usize>;

/// Sparsity of Jacobian and Hessian.
///
/// The sparsity mapping is produced by adding the constraints and the
/// objective to this structure dynamically.
#[derive(Debug, Default)]
pub(crate) struct Sparsity {
    pub(crate) jac_sp: JacSparsity,
    pub(crate) jac_cons_inds: Vec<Vec<usize>>,
    pub(crate) hes_sp: HesSparsity,
    pub(crate) hes_cons_inds: Vec<Vec<usize>>,
    pub(crate) hes_obj_inds: Vec<usize>,
}

impl Sparsity {
    pub(crate) fn new() -> Sparsity {
        Sparsity::default()
    }

    /// Get position of constraint and variable combination in Jacobian sparse
    /// representation.
    ///
    /// Creates a new Jacobian entry if it hasn't already been used.
    fn jac_index(&mut self, eid: (usize, ID)) -> usize {
        let id = self.jac_sp.len(); // incase need to create new
        *self.jac_sp.entry(eid).or_insert(id)
    }

    /// Get position of pair of variables in Hessian sparse representation.
    ///
    /// Creates a new Hessian entry if it hasn't already been used.
    fn hes_index(&mut self, eid: (ID, ID)) -> usize {
        let id = self.hes_sp.len(); // incase need to create new
        *self.hes_sp.entry(eid).or_insert(id)
    }

    /// Account for constraint in sparsity.
    ///
    /// The contributions to the Hessian and Jacobian sparsities are calculated.
    pub(crate) fn add_con(&mut self, expr: &Expression) {
        let cid = self.jac_cons_inds.len();
        let mut v_hes = Vec::new();
        let mut v_jac = Vec::new();
        match expr {
            Expression::ExprFix(e) => {
                v_jac.extend(e.d1_sparsity.iter().map(|Var(v)| self.jac_index((cid, *v))));
                v_hes.extend(
                    e.d2_sparsity
                        .iter()
                        .map(|(Var(v1), Var(v2))| self.hes_index((*v1, *v2))),
                );
            }
            Expression::ExprFixSum(es) => {
                for e in es {
                    v_jac.extend(e.d1_sparsity.iter().map(|Var(v)| self.jac_index((cid, *v))));
                    v_hes.extend(
                        e.d2_sparsity
                            .iter()
                            .map(|(Var(v1), Var(v2))| self.hes_index((*v1, *v2))),
                    );
                }
            }
            Expression::ExprDyn(e) => {
                v_jac.extend(e.info.lin.iter().map(|i| self.jac_index((cid, *i))));
                v_jac.extend(e.info.nlin.iter().map(|i| self.jac_index((cid, *i))));
                v_hes.extend(
                    e.info
                        .quad
                        .iter()
                        .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                );
                v_hes.extend(
                    e.info
                        .nquad
                        .iter()
                        .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                );
            }
            Expression::ExprDynSum(es) => {
                for e in es {
                    v_jac.extend(e.info.lin.iter().map(|i| self.jac_index((cid, *i))));
                    v_hes.extend(
                        e.info
                            .quad
                            .iter()
                            .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                    );
                }
                for e in es {
                    v_jac.extend(e.info.nlin.iter().map(|i| self.jac_index((cid, *i))));
                    v_hes.extend(
                        e.info
                            .nquad
                            .iter()
                            .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                    );
                }
            }
        }
        self.hes_cons_inds.push(v_hes);
        self.jac_cons_inds.push(v_jac);
    }

    /// Account for objective in sparsity.
    ///
    /// The objective only appears in the Hessian, and the gradient expects a
    /// dense reply with variables in order, so nothing needs to be additionally
    /// stored for that.
    pub(crate) fn add_obj(&mut self, expr: &Expression) {
        let mut v = Vec::new();
        match expr {
            Expression::ExprFix(e) => {
                v.extend(
                    e.d2_sparsity
                        .iter()
                        .map(|(Var(v1), Var(v2))| self.hes_index((*v1, *v2))),
                );
            }
            Expression::ExprFixSum(es) => {
                for e in es {
                    v.extend(
                        e.d2_sparsity
                            .iter()
                            .map(|(Var(v1), Var(v2))| self.hes_index((*v1, *v2))),
                    );
                }
            }
            Expression::ExprDyn(e) => {
                v.extend(
                    e.info
                        .quad
                        .iter()
                        .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                );
                v.extend(
                    e.info
                        .nquad
                        .iter()
                        .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                );
            }
            Expression::ExprDynSum(es) => {
                for e in es {
                    v.extend(
                        e.info
                            .quad
                            .iter()
                            .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                    );
                }
                for e in es {
                    v.extend(
                        e.info
                            .nquad
                            .iter()
                            .map(|(i, j)| self.hes_index((e.info.nlin[*i], e.info.nlin[*j]))),
                    );
                }
            }
        }
        self.hes_obj_inds = v;
    }

    pub(crate) fn jac_len(&self) -> usize {
        self.jac_sp.len()
    }

    pub(crate) fn hes_len(&self) -> usize {
        self.hes_sp.len()
    }
}
