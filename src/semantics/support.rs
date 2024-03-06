use crate::ppl::{Distribution, Event, Program, Statement, Var};
use crate::support::SupportSet;

use super::Transformer;

#[derive(Default)]
pub struct SupportTransformer;

impl Transformer for SupportTransformer {
    type Domain = Vec<SupportSet>;

    fn init(&mut self, program: &Program) -> Vec<SupportSet> {
        let num_vars = program.used_vars().num_vars();
        vec![SupportSet::zero(); num_vars]
    }

    fn transform_event(
        &mut self,
        _event: &Event,
        init: Vec<SupportSet>,
    ) -> (Vec<SupportSet>, Vec<SupportSet>) {
        (init.clone(), init) // TODO: make this more precise
    }

    fn transform_statement(&mut self, stmt: &Statement, init: Vec<SupportSet>) -> Vec<SupportSet> {
        if init.iter().all(SupportSet::is_empty) {
            return init;
        }
        match stmt {
            Statement::Sample {
                var,
                distribution,
                add_previous_value,
            } => Self::transform_distribution(distribution, *var, init, *add_previous_value),
            Statement::Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                let mut new_support = init[var.id()].clone();
                if !*add_previous_value {
                    new_support = SupportSet::zero();
                }
                if let Some((factor, w)) = addend {
                    new_support += init[w.id()].clone() * factor.0;
                }
                new_support += SupportSet::point(offset.0);
                let mut res = init;
                res[var.id()] = new_support;
                res
            }
            Statement::Decrement { var, offset } => {
                let mut res = init;
                res[var.id()] = res[var.id()].saturating_sub(offset.0);
                res
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = self.transform_event(cond, init);
                let then_res = self.transform_statements(then, then_res);
                let else_res = self.transform_statements(els, else_res);
                join_var_infos(&else_res, &then_res)
            }
            Statement::Fail => vec![],
            Statement::Normalize { given_vars, stmts } => {
                self.transform_normalize(given_vars, stmts, init)
            }
        }
    }
}

impl SupportTransformer {
    pub fn transform_distribution(
        dist: &Distribution,
        v: Var,
        init: Vec<SupportSet>,
        add_previous_value: bool,
    ) -> Vec<SupportSet> {
        let mut result = init.clone();
        if v.id() == result.len() {
            result.push(SupportSet::zero());
        }
        assert!(v.id() < result.len());
        if !add_previous_value {
            result[v.id()] = SupportSet::zero();
        }
        result[v.id()] += dist.support();
        result
    }

    pub fn transform_normalize(
        &mut self,
        given_vars: &[Var],
        block: &[Statement],
        var_info: Vec<SupportSet>,
    ) -> Vec<SupportSet> {
        if given_vars.is_empty() {
            self.transform_statements(block, var_info)
        } else {
            let v = given_vars[0];
            let rest = &given_vars[1..];
            let support = var_info[v.id()].clone();
            let range = support.finite_nonempty_range().unwrap_or_else(|| panic!("Cannot normalize with respect to variable `{v}`, because its value could not be proven to be bounded."));
            let mut joined = Vec::new();
            for i in range {
                let mut new_var_info = var_info.clone();
                new_var_info[v.id()] = SupportSet::from(i);
                let result = self.transform_normalize(rest, block, new_var_info);
                joined = join_var_infos(&joined, &result);
            }
            joined
        }
    }
}

pub fn join_var_infos(
    var_info_then: &[SupportSet],
    var_info_else: &[SupportSet],
) -> Vec<SupportSet> {
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
