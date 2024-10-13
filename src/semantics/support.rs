use std::ops::Index;

use crate::ppl::{Distribution, Event, Program, Statement, Var};
use crate::support::SupportSet;

use super::Transformer;

#[derive(Clone, Debug, PartialEq)]
pub enum VarSupport {
    Empty(usize),
    Prod(Vec<SupportSet>),
}

impl From<Vec<SupportSet>> for VarSupport {
    fn from(supports: Vec<SupportSet>) -> Self {
        let mut res = VarSupport::Prod(supports);
        res.normalize();
        res
    }
}

impl Index<&Var> for VarSupport {
    type Output = SupportSet;

    fn index(&self, v: &Var) -> &Self::Output {
        &self[*v]
    }
}

impl Index<Var> for VarSupport {
    type Output = SupportSet;

    fn index(&self, v: Var) -> &Self::Output {
        match self {
            VarSupport::Empty(_) => &SupportSet::Empty,
            VarSupport::Prod(s) => &s[v.id()],
        }
    }
}

impl std::fmt::Display for VarSupport {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VarSupport::Empty(_) => write!(f, "empty"),
            VarSupport::Prod(supports) => {
                let mut first = true;
                for support in supports {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "{support}")?;
                }
                Ok(())
            }
        }
    }
}

impl VarSupport {
    pub(crate) fn empty(num_vars: usize) -> VarSupport {
        VarSupport::Empty(num_vars)
    }

    pub(crate) fn zero(count: usize) -> VarSupport {
        VarSupport::Prod(vec![SupportSet::zero(); count])
    }

    pub(crate) fn as_vec(&self) -> Option<&[SupportSet]> {
        match self {
            VarSupport::Empty(_) => None,
            VarSupport::Prod(v) => Some(v),
        }
    }

    pub fn num_vars(&self) -> usize {
        match self {
            VarSupport::Empty(len) => *len,
            VarSupport::Prod(v) => v.len(),
        }
    }

    pub(crate) fn push(&mut self, support: SupportSet) {
        match self {
            VarSupport::Empty(n) => *n += 1,
            VarSupport::Prod(v) => v.push(support),
        }
    }

    fn normalize(&mut self) {
        let should_set_empty = match self {
            VarSupport::Empty(_) => false,
            VarSupport::Prod(v) => v.iter().any(SupportSet::is_empty),
        };
        if should_set_empty {
            let len = self.num_vars();
            let _ = std::mem::replace(self, Self::empty(len));
        }
    }

    pub(crate) fn is_subset_of(&self, other: &VarSupport) -> bool {
        match (self, other) {
            (VarSupport::Empty(_), _) => true,
            (_, VarSupport::Empty(_)) => false,
            (VarSupport::Prod(this), VarSupport::Prod(other)) => {
                debug_assert_eq!(this.len(), other.len());
                this.iter().zip(other).all(|(x, y)| x.is_subset_of(y))
            }
        }
    }

    pub(crate) fn join(&self, other: &VarSupport) -> VarSupport {
        match (self, other) {
            (VarSupport::Empty(_), res) | (res, VarSupport::Empty(_)) => res.clone(),
            (VarSupport::Prod(this), VarSupport::Prod(other)) => {
                let mut var_info = Vec::new();
                let len = this.len();
                debug_assert_eq!(other.len(), len);
                for i in 0..len {
                    match (this.get(i), other.get(i)) {
                        (Some(set), None) | (None, Some(set)) => var_info.push(set.clone()),
                        (Some(set1), Some(set2)) => {
                            var_info.push(set1.join(set2));
                        }
                        (None, None) => panic!("This should not happen"),
                    }
                }
                VarSupport::from(var_info)
            }
        }
    }

    pub(crate) fn update(&mut self, Var(v): Var, f: impl FnOnce(&mut SupportSet)) {
        match self {
            VarSupport::Empty(_) => {}
            VarSupport::Prod(supports) => {
                f(&mut supports[v]);
            }
        }
        self.normalize();
    }

    pub(crate) fn set(&mut self, var: Var, new: SupportSet) {
        self.update(var, |s| *s = new);
    }
}

#[derive(Default)]
pub struct SupportTransformer {
    unroll: usize,
}

impl SupportTransformer {
    pub(crate) fn with_unroll(mut self, unroll: usize) -> Self {
        self.unroll = unroll;
        self
    }
}

impl Transformer for SupportTransformer {
    type Domain = VarSupport;

    fn init(&mut self, program: &Program) -> VarSupport {
        VarSupport::zero(program.used_vars().num_vars())
    }

    fn transform_event(&mut self, event: &Event, init: VarSupport) -> (VarSupport, VarSupport) {
        match event {
            Event::InSet(v, set) => {
                let mut then_support = init.clone();
                then_support.update(*v, |s| s.retain_only(set.iter().map(|n| n.0)));
                let mut else_support = init;
                else_support.update(*v, |s| s.remove_all(set.iter().map(|n| n.0)));
                (then_support, else_support)
            }
            // TODO: make the approximation more precise for VarComparison
            Event::DataFromDist(..) | Event::VarComparison(..) => (init.clone(), init),
            Event::Complement(event) => {
                let (then_support, else_support) = self.transform_event(event, init);
                (else_support, then_support)
            }
            Event::Intersection(events) => {
                let mut else_support = VarSupport::empty(init.num_vars());
                let mut then_support = init;
                for event in events {
                    let (new_then, new_else) = self.transform_event(event, then_support);
                    then_support = new_then;
                    else_support = else_support.join(&new_else);
                }
                (then_support, else_support)
            }
        }
    }

    fn transform_statement(&mut self, stmt: &Statement, init: VarSupport) -> VarSupport {
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
                let mut new_support = init[var].clone();
                if !*add_previous_value {
                    new_support = SupportSet::zero();
                }
                if let Some((factor, w)) = addend {
                    new_support += init[w].clone() * factor.0;
                }
                new_support += SupportSet::point(offset.0);
                let mut res = init;
                res.set(*var, new_support);
                res
            }
            Statement::Decrement { var, offset } => {
                let mut res = init;
                res.update(*var, |s| *s = s.saturating_sub(offset.0));
                res
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = self.transform_event(cond, init);
                let then_res = self.transform_statements(then, then_res);
                let else_res = self.transform_statements(els, else_res);
                then_res.join(&else_res)
            }
            Statement::While { cond, body, unroll } => {
                let unroll_count = unroll.unwrap_or(self.unroll);
                let unroll_result = self.find_unroll_fixpoint(cond, body, init.clone());
                let unroll_count = if let Some((iters, _, _)) = unroll_result {
                    unroll_count.max(iters)
                } else {
                    unroll_count
                };
                let mut pre_loop = init;
                let mut rest = VarSupport::empty(pre_loop.num_vars());
                for _ in 0..unroll_count {
                    let (new_pre_loop, loop_exit) =
                        self.one_iteration(pre_loop.clone(), body, cond);
                    rest = rest.join(&loop_exit);
                    pre_loop = new_pre_loop;
                }
                let invariant = self.find_while_invariant(cond, body, pre_loop);
                let (_, loop_exit) = self.transform_event(cond, invariant.clone());
                rest.join(&loop_exit)
            }
            Statement::Fail => VarSupport::empty(init.num_vars()),
        }
    }
}

impl SupportTransformer {
    pub(crate) fn transform_distribution(
        dist: &Distribution,
        v: Var,
        init: VarSupport,
        add_previous_value: bool,
    ) -> VarSupport {
        let mut result = init.clone();
        if v.id() == result.num_vars() {
            result.push(SupportSet::zero());
        }
        assert!(v.id() < result.num_vars());
        if !add_previous_value {
            result.set(v, SupportSet::zero());
        }
        result.update(v, |s| *s += dist.support());
        result
    }

    pub(crate) fn find_unroll_fixpoint(
        &mut self,
        cond: &Event,
        body: &[Statement],
        init: VarSupport,
    ) -> Option<(usize, VarSupport, VarSupport)> {
        let mut pre_loop = init;
        let mut rest = VarSupport::empty(pre_loop.num_vars());
        // TODO: upper bound of this for loop should be the highest constant occurring in the loop or something like that
        for i in 0..100 {
            let (new_pre_loop, loop_exit) = self.one_iteration(pre_loop.clone(), body, cond);
            rest = rest.join(&loop_exit);
            if pre_loop == new_pre_loop {
                return Some((i, pre_loop, rest));
            }
            pre_loop = new_pre_loop;
        }
        None
    }

    pub(crate) fn find_while_invariant(
        &mut self,
        cond: &Event,
        body: &[Statement],
        init: VarSupport,
    ) -> VarSupport {
        let mut pre_loop = init;
        // Try to widen using join a few times
        // TODO: upper bound of this for loop should be the highest constant occurring in the loop or something like that
        for _ in 0..100 {
            let (new_pre_loop, _) = self.one_iteration(pre_loop.clone(), body, cond);
            if new_pre_loop.is_subset_of(&pre_loop) {
                return pre_loop;
            }
            pre_loop = pre_loop.join(&new_pre_loop);
        }
        // If widening with `join` did not work, use an actual widening operation.
        // The number of widening steps needed is at most twice the number of variables,
        // because each variable can be widened at most twice (once in each direction).
        for _ in 0..=2 * pre_loop.num_vars() {
            let (new_pre_loop, _) = self.one_iteration(pre_loop.clone(), body, cond);
            if new_pre_loop.is_subset_of(&pre_loop) {
                return pre_loop;
            }
            for v in 0..pre_loop.num_vars() {
                let v = Var(v);
                pre_loop.set(v, Self::widen(&pre_loop[v], &new_pre_loop[v]));
            }
        }
        let (new_pre_loop, _) = self.one_iteration(pre_loop.clone(), body, cond);
        assert!(new_pre_loop.is_subset_of(&pre_loop), "Widening failed.");
        pre_loop
    }

    fn widen(cur: &SupportSet, new: &SupportSet) -> SupportSet {
        match (cur, new) {
            (
                SupportSet::Range { start, end },
                SupportSet::Range {
                    start: new_start,
                    end: new_end,
                },
            ) => {
                let start = if start <= new_start { *start } else { 0 };
                let end = match (end, new_end) {
                    (Some(end), Some(new_end)) if new_end <= end => Some(*end),
                    _ => None,
                };
                SupportSet::Range { start, end }
            }
            _ => panic!("Cannot widen non-range supports"),
        }
    }

    fn one_iteration(
        &mut self,
        init: VarSupport,
        body: &[Statement],
        cond: &Event,
    ) -> (VarSupport, VarSupport) {
        let (enter, exit) = self.transform_event(cond, init);
        let post = self.transform_statements(body, enter);
        (post, exit)
    }
}
