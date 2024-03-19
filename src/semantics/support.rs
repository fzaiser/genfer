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

impl VarSupport {
    pub fn empty(num_vars: usize) -> VarSupport {
        VarSupport::Empty(num_vars)
    }

    pub fn zero(count: usize) -> VarSupport {
        VarSupport::Prod(vec![SupportSet::zero(); count])
    }

    pub fn as_vec(&self) -> Option<&[SupportSet]> {
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

    pub fn push(&mut self, support: SupportSet) {
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

    pub fn is_subset_of(&self, other: &VarSupport) -> bool {
        match (self, other) {
            (VarSupport::Empty(_), _) => true,
            (_, VarSupport::Empty(_)) => false,
            (VarSupport::Prod(this), VarSupport::Prod(other)) => {
                debug_assert_eq!(this.len(), other.len());
                this.iter().zip(other).all(|(x, y)| x.is_subset_of(y))
            }
        }
    }

    pub fn join(&self, other: &VarSupport) -> VarSupport {
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

    pub fn update(&mut self, Var(v): Var, f: impl FnOnce(&mut SupportSet)) {
        match self {
            VarSupport::Empty(_) => {}
            VarSupport::Prod(supports) => {
                f(&mut supports[v]);
            }
        }
        self.normalize();
    }

    pub fn set(&mut self, var: Var, new: SupportSet) {
        self.update(var, |s| *s = new);
    }
}

#[derive(Default)]
pub struct SupportTransformer;

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
            Event::DataFromDist(..) => (init.clone(), init),
            Event::VarComparison(..) => (init.clone(), init), // TODO: make this approximation more precise
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
            Statement::While { cond, body, .. } => {
                let (_, _loop_entry, loop_exit) = self.analyze_while(cond, body, init);
                loop_exit
            }
            Statement::Fail => VarSupport::empty(init.num_vars()),
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

    pub fn analyze_while(
        &mut self,
        cond: &Event,
        body: &[Statement],
        init: VarSupport,
    ) -> (Option<usize>, VarSupport, VarSupport) {
        let (mut loop_entry, mut loop_exit) = self.transform_event(cond, init);
        let mut loop_entry_invariant = loop_entry.clone();
        // TODO: upper bound of this for loop should be the highest constant occurring in the loop or something like that
        for i in 0..100 {
            let (new_loop_entry, new_loop_exit) =
                self.one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
            loop_entry_invariant = loop_entry_invariant.join(&new_loop_entry);
            if loop_entry == new_loop_entry && loop_exit == new_loop_exit {
                return (Some(i), loop_entry, loop_exit);
            }
            loop_entry = new_loop_entry;
            loop_exit = new_loop_exit;
        }
        // The number of widening steps needed is at most the number of variables:
        for _ in 0..=loop_entry.num_vars() {
            loop_entry_invariant = loop_entry_invariant.join(&loop_entry);
            let (new_loop_entry, new_loop_exit) =
                self.one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
            if new_loop_entry.is_subset_of(&loop_entry) && new_loop_exit.is_subset_of(&loop_exit) {
                assert_eq!(loop_exit, new_loop_exit);
                return (None, loop_entry_invariant, loop_exit);
            }
            for v in 0..loop_entry.num_vars() {
                let v = Var(v);
                match (&loop_entry[v], &new_loop_entry[v]) {
                    (
                        SupportSet::Range { start, end },
                        SupportSet::Range {
                            start: new_start,
                            end: new_end,
                        },
                    ) => {
                        if end.is_some() && new_end.is_none() {
                            unreachable!();
                        }
                        #[allow(clippy::manual_assert)]
                        if (new_end.is_some() && new_end < end) || new_start < start {
                            panic!("More iterations needed");
                        }
                        if new_end > end {
                            loop_entry.set(v, (*start..).into());
                        }
                    }
                    _ => {
                        dbg!(v, &loop_entry[v], &new_loop_entry[v]);
                        unreachable!("Unexpected variable supports")
                    }
                }
                match (&loop_exit[v], &new_loop_exit[v]) {
                    (
                        SupportSet::Range { start, end },
                        SupportSet::Range {
                            start: new_start,
                            end: new_end,
                        },
                    ) => {
                        if end.is_some() && new_end.is_none() {
                            unreachable!();
                        }
                        #[allow(clippy::manual_assert)]
                        if (new_end.is_some() && new_end < end) || new_start < start {
                            panic!("More iterations needed");
                        }
                        if new_end > end {
                            loop_exit.set(v, (*start..).into());
                        }
                    }
                    _ => unreachable!("Unexpected variable supports"),
                }
            }
        }
        let (new_loop_entry, new_loop_exit) =
            self.one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
        assert!(
            new_loop_entry.is_subset_of(&loop_entry) && new_loop_exit.is_subset_of(&loop_exit),
            "Widening failed."
        );
        (None, loop_entry_invariant, loop_exit)
    }

    fn one_iteration(
        &mut self,
        loop_entry: VarSupport,
        loop_exit: VarSupport,
        body: &[Statement],
        cond: &Event,
    ) -> (VarSupport, VarSupport) {
        let after_loop = self.transform_statements(body, loop_entry);
        let (repeat_supports, exit_supports) = self.transform_event(cond, after_loop);
        let loop_exit = loop_exit.join(&exit_supports);
        (repeat_supports, loop_exit)
    }

    pub fn transform_normalize(
        &mut self,
        given_vars: &[Var],
        block: &[Statement],
        var_info: VarSupport,
    ) -> VarSupport {
        if given_vars.is_empty() {
            self.transform_statements(block, var_info)
        } else {
            let v = given_vars[0];
            let rest = &given_vars[1..];
            let support = var_info[v].clone();
            let range = support.finite_nonempty_range().unwrap_or_else(|| panic!("Cannot normalize with respect to variable `{v}`, because its value could not be proven to be bounded."));
            let mut joined = VarSupport::empty(var_info.num_vars());
            for i in range {
                let mut new_var_info = var_info.clone();
                new_var_info.set(v, SupportSet::from(i));
                let result = self.transform_normalize(rest, block, new_var_info);
                joined = joined.join(&result);
            }
            joined
        }
    }
}
