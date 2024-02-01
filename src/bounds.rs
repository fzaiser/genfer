use std::{
    ops::{AddAssign, SubAssign},
    time::Duration,
};

use good_lp::{variable, ProblemVariables, Solution, SolverModel};
use ndarray::{arr0, indices, ArrayD, ArrayViewD, ArrayViewMutD, Axis, Dimension, Slice};
use num_traits::{One, Zero};

use crate::{
    multivariate_taylor::TaylorPoly,
    number::{Number, F64},
    ppl::{Distribution, Event, Natural, Program, Statement, Var},
    support::SupportSet,
};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
}

pub struct BoundCtx {
    program_var_count: usize,
    sym_var_count: usize,
    // Variables used nonlinearly, in [0,1)
    nonlinear_param_vars: Vec<usize>,
    constraints: Vec<SymConstraint>,
    soft_constraints: Vec<SymConstraint>,
}

impl BoundCtx {
    pub fn new() -> Self {
        Self {
            program_var_count: 0,
            sym_var_count: 0,
            nonlinear_param_vars: Vec::new(),
            constraints: Vec::new(),
            soft_constraints: Vec::new(),
        }
    }

    pub fn sym_var_count(&self) -> usize {
        self.sym_var_count
    }

    pub fn constraints(&self) -> &[SymConstraint] {
        &self.constraints
    }

    pub fn soft_constraints(&self) -> &[SymConstraint] {
        &self.soft_constraints
    }

    pub fn fresh_sym_var_idx(&mut self) -> usize {
        let var = self.sym_var_count;
        self.sym_var_count += 1;
        var
    }

    pub fn fresh_sym_var(&mut self) -> SymExpr {
        SymExpr::var(self.fresh_sym_var_idx())
    }

    pub fn add_constraint(&mut self, constraint: SymConstraint) {
        // println!("Adding constraint {constraint}");
        self.constraints.push(constraint);
    }

    pub fn add_soft_constraint(&mut self, constraint: SymConstraint) {
        // println!("Adding soft constraint {constraint}");
        self.soft_constraints.push(constraint);
    }

    pub fn new_geo_param_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx();
        let var = SymExpr::var(idx);
        self.nonlinear_param_vars.push(idx);
        self.add_constraint(var.clone().must_ge(SymExpr::zero()));
        self.add_constraint(var.clone().must_le(SymExpr::one()));
        self.add_soft_constraint(var.clone().must_lt(SymExpr::one()));
        var
    }

    pub fn add_bounds(&mut self, lhs: GeometricBound, rhs: GeometricBound) -> GeometricBound {
        let count = self.program_var_count;
        let mut geo_params = Vec::with_capacity(count);
        for i in 0..count {
            let a = lhs.geo_params[i].clone();
            let b = rhs.geo_params[i].clone();
            if a == b {
                geo_params.push(a);
            } else {
                let new_geo_param_var = self.new_geo_param_var();
                geo_params.push(new_geo_param_var.clone());
                self.add_constraint(a.clone().must_le(new_geo_param_var.clone()));
                self.add_constraint(b.clone().must_le(new_geo_param_var.clone()));
                // The following constraint is not necessary, but it helps the solver:
                // We require the new variable to be the maximum of the other two.
                self.add_constraint(SymConstraint::or(vec![
                    new_geo_param_var.clone().must_eq(a),
                    new_geo_param_var.must_eq(b),
                ]));
            }
        }
        let polynomial = lhs.polynomial + rhs.polynomial; // TODO: check whether this is correct
        GeometricBound {
            polynomial,
            geo_params,
        }
    }

    pub fn add_bound_results(&mut self, lhs: BoundResult, rhs: BoundResult) -> BoundResult {
        let bound = self.add_bounds(lhs.bound, rhs.bound);
        let var_supports = SupportSet::join_vecs(&lhs.var_supports, &rhs.var_supports);
        BoundResult {
            bound,
            var_supports,
        }
    }

    // TODO avoid code duplication with `bound_event`
    pub fn var_supports_event(
        init: Vec<SupportSet>,
        event: &Event,
    ) -> (Vec<SupportSet>, Vec<SupportSet>) {
        match event {
            Event::InSet(v, set) => {
                let mut then_support = init.clone();
                then_support[v.id()].retain_only(set.iter().map(|n| n.0));
                if then_support[v.id()].is_empty() {
                    for w in 0..then_support.len() {
                        then_support[w] = SupportSet::empty();
                    }
                }
                let mut else_support = init;
                else_support[v.id()].remove_all(set.iter().map(|n| n.0));
                if else_support[v.id()].is_empty() {
                    for w in 0..else_support.len() {
                        else_support[w] = SupportSet::empty();
                    }
                }
                (then_support, else_support)
            }
            Event::DataFromDist(..) => (init.clone(), init),
            Event::VarComparison(..) => todo!(),
            Event::Complement(event) => {
                let (then_support, else_support) = Self::var_supports_event(init, event);
                (else_support, then_support)
            }
            Event::Intersection(events) => {
                let mut else_support = vec![SupportSet::empty(); init.len()];
                let mut then_support = init;
                for event in events {
                    let (new_then, new_else) = Self::var_supports_event(then_support, event);
                    then_support = new_then;
                    else_support = SupportSet::join_vecs(&new_else, &else_support);
                }
                (then_support, else_support)
            }
        }
    }

    // TODO: avoid code duplication with `bound_statement`
    pub fn var_supports_statement(init: Vec<SupportSet>, stmt: &Statement) -> Vec<SupportSet> {
        if init.iter().all(|s| s.is_empty()) {
            return init;
        }
        match stmt {
            Statement::Sample {
                var,
                distribution,
                add_previous_value,
            } => {
                let mut res = init;
                if !*add_previous_value {
                    res[var.id()] = SupportSet::zero();
                }
                match distribution {
                    Distribution::Bernoulli(_) => {
                        res[var.id()] += (0..2).into();
                        res
                    }
                    Distribution::Geometric(_) if !add_previous_value => {
                        res[var.id()] += SupportSet::naturals();
                        res
                    }
                    Distribution::Uniform { start, end } => {
                        res[var.id()] += (start.0..end.0).into();
                        res
                    }
                    _ => todo!(),
                }
            }
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
            Statement::Decrement { var, amount } => {
                let mut res = init;
                res[var.id()] = res[var.id()].saturating_sub(amount.0);
                res
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = Self::var_supports_event(init, cond);
                let then_res = Self::var_supports_statements(then_res, then);
                let else_res = Self::var_supports_statements(else_res, els);
                SupportSet::join_vecs(&else_res, &then_res)
            }
            Statement::Fail => vec![SupportSet::empty(); init.len()],
            Statement::Normalize { .. } => todo!(),
            Statement::While { cond, body, .. } => {
                let (_, _loop_entry, loop_exit) = Self::analyze_while_support(cond, body, init);
                loop_exit
            }
        }
    }

    fn analyze_while_support(
        cond: &Event,
        body: &[Statement],
        init: Vec<SupportSet>,
    ) -> (Option<usize>, Vec<SupportSet>, Vec<SupportSet>) {
        let (mut loop_entry, mut loop_exit) = Self::var_supports_event(init, cond);
        // TODO: upper bound of the loop should be the highest constant occurring in the loop or something like that
        for i in 0..100 {
            let (new_loop_entry, new_loop_exit) =
                Self::one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
            if loop_entry == new_loop_entry && loop_exit == new_loop_exit {
                return (Some(i + 1), loop_entry, loop_exit);
            }
            loop_entry = new_loop_entry;
            loop_exit = new_loop_exit;
        }
        // The number of widening steps needed is at most the number of variables:
        for _ in 0..loop_entry.len() + 1 {
            let (new_loop_entry, new_loop_exit) =
                Self::one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
            if SupportSet::vec_is_subset_of(&new_loop_entry, &loop_entry)
                && SupportSet::vec_is_subset_of(&new_loop_exit, &loop_exit)
            {
                assert_eq!(loop_exit, new_loop_exit);
                return (None, loop_entry, loop_exit);
            }
            for v in 0..loop_entry.len() {
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
                        if (new_end.is_some() && new_end < end) || new_start < start {
                            panic!("More iterations needed");
                        }
                        if new_end > end {
                            loop_entry[v] = (*start..).into();
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
                        if (new_end.is_some() && new_end < end) || new_start < start {
                            panic!("More iterations needed");
                        }
                        if new_end > end {
                            loop_exit[v] = (*start..).into();
                        }
                    }
                    _ => unreachable!("Unexpected variable supports"),
                }
            }
        }
        let (new_loop_entry, new_loop_exit) =
            Self::one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
        assert!(
            SupportSet::vec_is_subset_of(&new_loop_entry, &loop_entry)
                && SupportSet::vec_is_subset_of(&new_loop_exit, &loop_exit),
            "Widening failed."
        );
        (None, loop_entry, loop_exit)
    }

    fn one_iteration(
        loop_entry: Vec<SupportSet>,
        loop_exit: Vec<SupportSet>,
        body: &[Statement],
        cond: &Event,
    ) -> (Vec<SupportSet>, Vec<SupportSet>) {
        let after_loop = Self::var_supports_statements(loop_entry, body);
        let (repeat_supports, exit_supports) = Self::var_supports_event(after_loop, cond);
        let loop_exit = SupportSet::join_vecs(&exit_supports, &loop_exit);
        (repeat_supports, loop_exit)
    }

    pub fn var_supports_statements(init: Vec<SupportSet>, stmts: &[Statement]) -> Vec<SupportSet> {
        let mut cur = init;
        for stmt in stmts {
            cur = Self::var_supports_statement(cur, stmt);
        }
        cur
    }

    pub fn var_supports_program(program: &Program) -> Vec<SupportSet> {
        let mut supports = vec![SupportSet::zero(); program.used_vars().num_vars()];
        for stmt in &program.stmts {
            supports = Self::var_supports_statement(supports, stmt);
        }
        supports
    }

    #[allow(clippy::only_used_in_recursion)]
    pub fn bound_event(&mut self, init: BoundResult, event: &Event) -> (BoundResult, BoundResult) {
        match event {
            Event::InSet(v, set) => {
                let alpha = init.bound.geo_params[v.id()].clone();
                let one_minus_alpha_v =
                    SymPolynomial::one() - SymPolynomial::var(*v) * alpha.clone();
                let mut then_bound = GeometricBound {
                    polynomial: SymPolynomial::zero(),
                    geo_params: init.bound.geo_params.clone(),
                };
                then_bound.geo_params[v.id()] = SymExpr::zero();
                let mut then_res = BoundResult {
                    bound: then_bound,
                    var_supports: init.var_supports.clone(),
                };
                then_res.var_supports[v.id()].retain_only(set.iter().map(|n| n.0));
                if then_res.var_supports[v.id()].is_empty() {
                    for w in 0..then_res.var_supports.len() {
                        then_res.var_supports[w] = SupportSet::empty();
                    }
                }
                let mut else_res = init;
                else_res.var_supports[v.id()].remove_all(set.iter().map(|n| n.0));
                if else_res.var_supports[v.id()].is_empty() {
                    for w in 0..else_res.var_supports.len() {
                        else_res.var_supports[w] = SupportSet::empty();
                    }
                }
                for Natural(n) in set {
                    let mut coeff = SymPolynomial::zero();
                    for i in 0..=*n {
                        coeff += else_res.bound.polynomial.coeff_of_var_power(*v, i as usize)
                            * alpha.clone().pow((n - i) as i32);
                    }
                    let monomial = coeff.clone() * SymPolynomial::var_power(*v, *n);
                    then_res.bound.polynomial += monomial.clone();
                    else_res.bound.polynomial -= monomial.clone() * one_minus_alpha_v.clone();
                }
                (then_res, else_res)
            }
            Event::DataFromDist(data, dist) => {
                if let Distribution::Bernoulli(p) = dist {
                    let p_compl = p.complement();
                    let p = F64::from_ratio(p.numer, p.denom).into();
                    let p_compl = F64::from_ratio(p_compl.numer, p_compl.denom).into();
                    let (then, els) = match data.0 {
                        0 => (p_compl, p),
                        1 => (p, p_compl),
                        _ => (0.0, 1.0),
                    };
                    let mut then_res = init.clone();
                    let mut else_res = init;
                    then_res.bound.polynomial *= SymExpr::from(then);
                    else_res.bound.polynomial *= SymExpr::from(els);
                    (then_res, else_res)
                } else {
                    todo!()
                }
            }
            Event::VarComparison(..) => todo!(),
            Event::Complement(event) => {
                let (then_res, else_res) = self.bound_event(init, event);
                (else_res, then_res)
            }
            Event::Intersection(events) => {
                let mut else_res = BoundResult {
                    bound: GeometricBound::zero(init.bound.geo_params.len()),
                    var_supports: vec![SupportSet::empty(); init.var_supports.len()],
                };
                let mut then_res = init;
                for event in events {
                    let (new_then, new_else) = self.bound_event(then_res, event);
                    then_res = new_then;
                    else_res = self.add_bound_results(else_res, new_else);
                }
                (then_res, else_res)
            }
        }
    }

    pub fn bound_statement(&mut self, init: BoundResult, stmt: &Statement) -> BoundResult {
        match stmt {
            Statement::Sample {
                var,
                distribution,
                add_previous_value,
            } => {
                let mut res = if *add_previous_value {
                    init
                } else {
                    init.marginalize(*var)
                };
                match distribution {
                    Distribution::Bernoulli(p) => {
                        res.var_supports[var.id()] += (0..2).into();
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        res.bound.polynomial *=
                            SymPolynomial::var(*var) * SymExpr::from(p) + (1.0 - p).into();
                        res
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        res.var_supports[var.id()] += SupportSet::naturals();
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        res.bound.polynomial *= SymExpr::from(p);
                        res.bound.geo_params[var.id()] = SymExpr::from(1.0 - p);
                        res
                    }
                    Distribution::Uniform { start, end } => {
                        res.var_supports[var.id()] += (start.0..end.0).into();
                        let mut factor = SymPolynomial::zero();
                        let len = f64::from(end.0 - start.0);
                        for i in start.0..end.0 {
                            factor += SymPolynomial::var_power(*var, i) / len.into();
                        }
                        res.bound.polynomial *= factor;
                        res
                    }
                    _ => todo!(),
                }
            }
            Statement::Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                if let (None, Natural(n)) = (addend, offset) {
                    let mut new_bound = if *add_previous_value {
                        init
                    } else {
                        init.marginalize(*var)
                    };
                    new_bound.bound.polynomial *= SymPolynomial::var_power(*var, *n);
                    new_bound.var_supports[var.id()] += SupportSet::point(*n);
                    new_bound
                } else {
                    todo!()
                }
            }
            Statement::Decrement { var, amount } => {
                let mut cur = init;
                let alpha = cur.bound.geo_params[var.id()].clone();
                for _ in 0..amount.0 {
                    let polynomial = cur.bound.polynomial;
                    let (p0, shifted) = polynomial.extract_zero_and_shift_left(*var);
                    let other = p0
                        * (SymPolynomial::one()
                            + (SymPolynomial::one() - SymPolynomial::var(*var)) * alpha.clone());
                    cur.bound.polynomial = shifted + other;
                }
                cur.var_supports[var.id()] = cur.var_supports[var.id()].saturating_sub(amount.0);
                cur
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = self.bound_event(init, cond);
                let then_res = self.bound_statements(then_res, then);
                let else_res = self.bound_statements(else_res, els);
                self.add_bound_results(then_res, else_res)
            }
            Statement::Fail => BoundResult {
                bound: GeometricBound::zero(self.program_var_count),
                var_supports: vec![SupportSet::empty(); init.var_supports.len()],
            },
            Statement::Normalize { .. } => todo!(),
            Statement::While { cond, unroll, body } => {
                let mut pre_loop = init;
                let mut rest = BoundResult {
                    bound: GeometricBound::zero(self.program_var_count),
                    var_supports: vec![SupportSet::empty(); pre_loop.var_supports.len()],
                };
                let unroll_count = unroll.unwrap_or(0);
                let (iters, invariant_supports, _) =
                    Self::analyze_while_support(cond, body, pre_loop.var_supports.clone());
                let unroll_count = if let Some(iters) = iters {
                    unroll_count.max(iters)
                } else {
                    unroll_count
                };
                println!("Unrolling {unroll_count} times");
                for _ in 0..unroll_count {
                    let (then_bound, else_bound) = self.bound_event(pre_loop.clone(), cond);
                    pre_loop = self.bound_statements(then_bound, body);
                    rest = self.add_bound_results(rest, else_bound);
                }
                let shape = invariant_supports
                    .iter()
                    .map(|s| match s {
                        SupportSet::Empty => 0,
                        SupportSet::Range { end, .. } => {
                            if let Some(end) = end {
                                *end as usize + 1
                            } else {
                                1
                            }
                        }
                        _ => todo!(),
                    })
                    .collect::<Vec<_>>();
                let finite_supports = invariant_supports
                    .iter()
                    .map(|s| s.is_empty() || s.finite_nonempty_range().is_some())
                    .collect::<Vec<_>>();
                let (loop_entry, loop_exit) = self.bound_event(pre_loop, cond);
                rest = self.add_bound_results(rest, loop_exit);
                let invariant = BoundResult {
                    bound: self.new_bound(shape, &finite_supports),
                    var_supports: invariant_supports,
                };
                println!("Invariant: {}", invariant);
                self.assert_le(&loop_entry.bound, &invariant.bound);
                let idx = self.fresh_sym_var_idx();
                let c = SymExpr::var(idx);
                println!("Invariant-c: {c}");
                self.nonlinear_param_vars.push(idx);
                self.add_constraint(c.clone().must_ge(SymExpr::zero()));
                self.add_constraint(c.clone().must_le(SymExpr::one()));
                self.add_soft_constraint(c.clone().must_lt(SymExpr::one()));
                let mut cur_bound = invariant.clone();
                for stmt in body {
                    cur_bound = self.bound_statement(cur_bound, stmt);
                }
                let (post_loop, mut exit) = self.bound_event(cur_bound, cond);
                self.assert_le(&post_loop.bound, &(invariant.bound.clone() * c.clone()));
                exit.bound = exit.bound / (SymExpr::one() - c.clone());
                self.add_bound_results(exit, rest)
            }
        }
    }

    pub fn bound_statements(&mut self, init: BoundResult, stmts: &[Statement]) -> BoundResult {
        let mut cur = init;
        for stmt in stmts {
            cur = self.bound_statement(cur, stmt);
        }
        cur
    }

    pub fn bound_program(&mut self, program: &Program) -> BoundResult {
        self.program_var_count = program.used_vars().num_vars();
        let init_bound = GeometricBound {
            polynomial: SymPolynomial::one(),
            geo_params: vec![SymExpr::zero(); self.program_var_count],
        };
        let init = BoundResult {
            bound: init_bound,
            var_supports: vec![SupportSet::zero(); self.program_var_count],
        };
        self.bound_statements(init, &program.stmts)
    }

    pub fn output_python(&self, bound: &GeometricBound) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "import numpy as np").unwrap();
        writeln!(out, "from scipy.optimize import *").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "nvars = {}", self.sym_var_count).unwrap();
        writeln!(out, "bounds = Bounds(").unwrap();
        write!(out, "  [").unwrap();
        for i in 0..self.sym_var_count {
            if self.nonlinear_param_vars.contains(&i) {
                write!(out, "0, ").unwrap();
            } else {
                write!(out, "-np.inf, ").unwrap();
            }
        }
        writeln!(out, "],").unwrap();
        write!(out, "  [").unwrap();
        for i in 0..self.sym_var_count {
            if self.nonlinear_param_vars.contains(&i) {
                write!(out, "1, ").unwrap();
            } else {
                write!(out, "np.inf, ").unwrap();
            }
        }
        writeln!(out, "])").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "def constraint_fun(x):").unwrap();
        writeln!(out, "  return [").unwrap();
        for c in &self.constraints {
            match c {
                SymConstraint::Eq(_, _) => continue,
                SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                    writeln!(out, "    ({}) - ({}),", rhs.to_python(), lhs.to_python()).unwrap();
                }
                SymConstraint::Or(_) => continue,
            }
        }
        writeln!(out, "  ]").unwrap();
        writeln!(out, "constraints = NonlinearConstraint(constraint_fun, 0, np.inf, jac='2-point', hess=BFGS())").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "def objective(x):").unwrap();
        let numer = bound
            .polynomial
            .coeffs
            .iter()
            .fold(SymExpr::zero(), |acc, c| acc + c.clone());
        write!(out, "  return ({}) / (", numer.to_python()).unwrap();
        for e in &bound.geo_params {
            write!(out, "(1 - {}) * ", e.to_python()).unwrap();
        }
        writeln!(out, "1)").unwrap();
        writeln!(out, "x0 = np.full((nvars,), 0.9)").unwrap();
        out
    }

    pub fn output_python_z3(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "import z3").unwrap();
        writeln!(out, "z3.set_option(precision=5)").unwrap();
        writeln!(out).unwrap();
        for i in 0..self.sym_var_count {
            writeln!(out, "x{i} = Real('x{i}')").unwrap();
        }
        writeln!(out, "s = Solver()").unwrap();
        for constraint in &self.constraints {
            writeln!(out, "s.add({})", constraint.to_python_z3()).unwrap();
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "s.add({})", constraint.to_python_z3()).unwrap();
        }
        writeln!(out).unwrap();
        out
    }

    pub fn output_smt(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "(set-logic QF_NRA)").unwrap();
        writeln!(out, "(set-option :pp.decimal true)").unwrap();
        writeln!(out).unwrap();
        for i in 0..self.sym_var_count() {
            writeln!(out, "(declare-const {} Real)", SymExpr::var(i)).unwrap();
        }
        writeln!(out).unwrap();
        for constraint in &self.constraints {
            writeln!(out, "(assert {})", constraint).unwrap();
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "(assert-soft {})", constraint).unwrap();
        }
        writeln!(out).unwrap();
        writeln!(out, "(check-sat)").unwrap();
        writeln!(out, "(get-model)").unwrap();
        out
    }

    fn new_polynomial(&mut self, shape: Vec<usize>) -> SymPolynomial {
        let mut coeffs = ArrayD::zeros(shape);
        for c in &mut coeffs {
            *c = self.fresh_sym_var();
        }
        SymPolynomial::new(coeffs)
    }

    fn new_bound(&mut self, shape: Vec<usize>, finite_supports: &[bool]) -> GeometricBound {
        let mut geo_params = vec![SymExpr::zero(); shape.len()];
        for (v, p) in geo_params.iter_mut().enumerate() {
            if !finite_supports[v] {
                *p = self.new_geo_param_var();
            }
        }
        GeometricBound {
            polynomial: self.new_polynomial(shape),
            geo_params,
        }
    }

    fn assert_le_helper(
        &mut self,
        lhs_coeffs: &ArrayD<SymExpr>,
        lhs_geo_params: &[SymExpr],
        rhs_coeffs: &ArrayD<SymExpr>,
        rhs_geo_params: &[SymExpr],
    ) {
        if lhs_geo_params.is_empty() {
            assert!(lhs_coeffs.len() == 1);
            assert!(rhs_coeffs.len() == 1);
            let lhs_coeff = lhs_coeffs.first().unwrap().clone();
            let rhs_coeff = rhs_coeffs.first().unwrap().clone();
            self.add_constraint(lhs_coeff.must_le(rhs_coeff));
            return;
        }
        let len_lhs = if lhs_coeffs.ndim() == 0 {
            1
        } else {
            lhs_coeffs.len_of(Axis(0))
        };
        let len_rhs = if rhs_coeffs.ndim() == 0 {
            1
        } else {
            rhs_coeffs.len_of(Axis(0))
        };
        let len = len_lhs.max(len_rhs);
        let lhs_shape = lhs_coeffs.shape();
        let rhs_shape = rhs_coeffs.shape();
        let mut lhs_inequality = if lhs_coeffs.ndim() == 0 {
            ArrayD::zeros(&[][..])
        } else {
            ArrayD::zeros(&lhs_shape[1..])
        };
        let mut rhs_inequality = if rhs_coeffs.ndim() == 0 {
            ArrayD::zeros(&[][..])
        } else {
            ArrayD::zeros(&rhs_shape[1..])
        };
        for i in 0..len {
            lhs_inequality.mapv_inplace(|c| c * lhs_geo_params[0].clone());
            rhs_inequality.mapv_inplace(|c| c * rhs_geo_params[0].clone());
            if lhs_coeffs.ndim() == 0 {
                if i == 0 {
                    *lhs_inequality.first_mut().unwrap() += lhs_coeffs.first().unwrap().clone();
                }
            } else if i < len_lhs {
                lhs_inequality += &lhs_coeffs.index_axis(Axis(0), i);
            }
            if rhs_coeffs.ndim() == 0 {
                if i == 0 {
                    *rhs_inequality.first_mut().unwrap() += rhs_coeffs.first().unwrap().clone();
                }
            } else if i < len_rhs {
                rhs_inequality += &rhs_coeffs.index_axis(Axis(0), i);
            }
            self.assert_le_helper(
                &lhs_inequality,
                &lhs_geo_params[1..],
                &rhs_inequality,
                &rhs_geo_params[1..],
            );
        }
    }

    fn assert_le(&mut self, lhs: &GeometricBound, rhs: &GeometricBound) {
        // println!("Asserting less than:\n{lhs}\n<=\n{rhs}");
        for i in 0..self.program_var_count {
            self.add_constraint(lhs.geo_params[i].clone().must_le(rhs.geo_params[i].clone()));
        }
        self.assert_le_helper(
            &lhs.polynomial.coeffs,
            &lhs.geo_params,
            &rhs.polynomial.coeffs,
            &rhs.geo_params,
        );
    }

    fn solve_lp_for_nonlinear_var_assignment(
        &self,
        bound: &GeometricBound,
        nonlinear_var_assignment: &[f64],
    ) -> Result<GeometricBound, SolverError> {
        let mut replacements = (0..self.sym_var_count())
            .map(SymExpr::var)
            .collect::<Vec<_>>();
        for (i, v) in self.nonlinear_param_vars.iter().enumerate() {
            replacements[*v] = nonlinear_var_assignment[i].into();
        }
        let bound = bound.substitute(&replacements);
        let constraints = self
            .constraints
            .iter()
            .map(|c| c.substitute(&replacements))
            .collect::<Vec<_>>();
        let linear_constraints = constraints
            .iter()
            .map(|constraint| {
                constraint.extract_linear().unwrap_or_else(|| {
                    panic!("Constraint is not linear in the program variables: {constraint}")
                })
            })
            .collect::<Vec<_>>();
        let mut lp = ProblemVariables::new();
        let mut var_list = Vec::new();
        for replacement in replacements {
            match replacement {
                SymExpr::Variable(_) => {
                    var_list.push(lp.add_variable());
                }
                SymExpr::Constant(c) => {
                    var_list.push(lp.add(variable().min(c).max(c)));
                }
                _ => unreachable!(),
            }
        }
        let objective = bound
            .polynomial
            .coeffs
            .iter()
            .fold(good_lp::Expression::from(0.), |acc, c| {
                acc + c.extract_linear().unwrap().to_lp_expr(&var_list)
            });
        let mut problem = lp.minimise(objective).using(good_lp::default_solver);
        for constraint in &linear_constraints {
            problem.add_constraint(constraint.to_lp_constraint(&var_list));
        }
        let solution = problem.solve().map_err(|err| match err {
            good_lp::ResolutionError::Unbounded => panic!("Optimal solution is unbounded."),
            good_lp::ResolutionError::Infeasible => SolverError::Infeasible,
            good_lp::ResolutionError::Other(msg) => todo!("Other error: {msg}"),
            good_lp::ResolutionError::Str(msg) => todo!("Error: {msg}"),
        })?;
        let solution = var_list
            .iter()
            .map(|v| solution.value(*v))
            .collect::<Vec<_>>();
        let mut resolved_bound = bound;
        for coeff in &mut resolved_bound.polynomial.coeffs {
            let val = coeff.eval(&solution);
            *coeff = SymExpr::Constant(val);
        }
        for geo_param in &mut resolved_bound.geo_params {
            let val = geo_param.eval(&solution);
            *geo_param = SymExpr::Constant(val);
        }
        Ok(resolved_bound)
    }

    pub fn solve_z3(
        &self,
        bound: &GeometricBound,
        timeout: Duration,
        optimize: bool,
    ) -> Result<GeometricBound, SolverError> {
        fn z3_real_to_f64(real: &z3::ast::Real) -> Option<f64> {
            if let Some((n, d)) = real.as_real() {
                return Some(n as f64 / d as f64);
            }
            let string = real.to_string();
            if let Ok(f) = string.parse::<f64>() {
                Some(f)
            } else {
                let words = string.split_whitespace().collect::<Vec<_>>();
                if words.len() == 3 && words[0] == "(/" && words[2].ends_with(')') {
                    let n = words[1].parse::<f64>().ok()?;
                    let d = words[2][..words[2].len() - 1].parse::<f64>().ok()?;
                    return Some(n / d);
                }
                None
            }
        }
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for constraint in self.constraints() {
            solver.assert(&constraint.to_z3(&ctx))
        }
        for constraint in self.soft_constraints() {
            solver.assert(&constraint.to_z3(&ctx))
        }
        match solver.check() {
            z3::SatResult::Unknown => {
                if let Some(reason) = solver.get_reason_unknown() {
                    if reason == "timeout" {
                        return Err(SolverError::Timeout);
                    }
                    panic!("Solver responded 'unknown': {reason}")
                } else {
                    panic!("Solver responded 'unknown' but no reason was given.")
                }
            }
            z3::SatResult::Unsat => return Err(SolverError::Infeasible),
            z3::SatResult::Sat => {}
        }
        let objective = bound.total_mass();
        let mut obj_lo = 0.0;
        let mut obj_hi;
        let model = loop {
            let model = solver
                .get_model()
                .unwrap_or_else(|| panic!("SMT solver's model is not available"));
            for var in 0..self.sym_var_count() {
                let val = model
                    .eval(&z3::ast::Real::new_const(&ctx, var as u32), false)
                    .unwrap();
                let val = z3_real_to_f64(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                println!("{var} -> {val}", var = SymExpr::var(var));
            }
            let mut resolved_bound = bound.clone();
            for coeff in &mut resolved_bound.polynomial.coeffs {
                let val = model.eval(&coeff.to_z3(&ctx), false).unwrap();
                let val = z3_real_to_f64(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                *coeff = SymExpr::Constant(val);
            }
            for geo_param in &mut resolved_bound.geo_params {
                let val = model.eval(&geo_param.to_z3(&ctx), false).unwrap();
                let val = z3_real_to_f64(&val)
                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                *geo_param = SymExpr::Constant(val);
            }
            println!("SMT solution:\n {resolved_bound}");
            let obj_val =
                z3_real_to_f64(&model.eval(&objective.to_z3(&ctx), false).unwrap()).unwrap();
            println!("Total mass (objective): {obj_val}");
            obj_hi = obj_val;
            println!("Objective bound: [{obj_lo}, {obj_hi}]");
            if !optimize || obj_hi - obj_lo < 0.1 * obj_hi {
                break model;
            }
            solver.push();
            let mid = (obj_lo + obj_hi) / 2.0;
            solver.assert(&objective.to_z3(&ctx).le(&SymExpr::from(mid).to_z3(&ctx)));
            loop {
                match solver.check() {
                    z3::SatResult::Sat => {
                        break;
                    }
                    z3::SatResult::Unknown | z3::SatResult::Unsat => {
                        solver.pop(1);
                        obj_lo = mid;
                    }
                }
            }
        };
        println!("Optimizing polynomial coefficients...");
        let nonlinear_var_assignment = self
            .nonlinear_param_vars
            .iter()
            .map(|var| {
                let val = model.eval(&SymExpr::var(*var).to_z3(&ctx), false).unwrap();
                z3_real_to_f64(&val).unwrap_or_else(|| panic!("{val} cannot be converted to f64"))
            })
            .collect::<Vec<_>>();
        let optimized_solution =
            self.solve_lp_for_nonlinear_var_assignment(bound, &nonlinear_var_assignment)?;
        Ok(optimized_solution)
    }
}

impl Default for BoundCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct BoundResult {
    pub bound: GeometricBound,
    pub var_supports: Vec<SupportSet>,
}

impl BoundResult {
    pub fn marginalize(self, var: Var) -> BoundResult {
        let mut var_supports = self.var_supports;
        if !var_supports[var.id()].is_empty() {
            var_supports[var.id()] = SupportSet::zero();
        }
        BoundResult {
            bound: self.bound.marginalize(var),
            var_supports,
        }
    }
}

impl std::fmt::Display for BoundResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, support) in self.var_supports.iter().enumerate() {
            writeln!(
                f,
                "Support of {var}: {support}",
                var = Var(i),
                support = support
            )?;
        }
        writeln!(f, "Bound:\n{bound}", bound = self.bound)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometricBound {
    polynomial: SymPolynomial,
    geo_params: Vec<SymExpr>,
}

impl GeometricBound {
    fn zero(n: usize) -> Self {
        GeometricBound {
            polynomial: SymPolynomial::zero(),
            geo_params: vec![SymExpr::zero(); n],
        }
    }

    pub fn marginalize(&self, var: Var) -> Self {
        let mut polynomial = self.polynomial.clone();
        let mut geo_params = self.geo_params.clone();
        polynomial =
            polynomial.marginalize(var) * (SymExpr::one() - geo_params[var.id()].clone()).inverse();
        geo_params[var.id()] = SymExpr::zero();
        Self {
            polynomial,
            geo_params,
        }
    }

    fn substitute(&self, replacements: &[SymExpr]) -> GeometricBound {
        Self {
            polynomial: self.polynomial.substitute(replacements),
            geo_params: self
                .geo_params
                .iter()
                .map(|p| p.substitute(replacements))
                .collect(),
        }
    }

    pub fn evaluate_var<T: From<f64> + Number>(
        &self,
        inputs: &[T],
        var: Var,
        degree_p1: usize,
    ) -> TaylorPoly<T> {
        let vars = inputs
            .iter()
            .enumerate()
            .map(|(w, val)| {
                if w == var.id() {
                    TaylorPoly::var(var, val.clone(), degree_p1)
                } else {
                    TaylorPoly::from(val.clone())
                }
            })
            .collect::<Vec<_>>();
        self.eval(&vars)
    }

    fn eval<T: From<f64> + Number>(&self, inputs: &[TaylorPoly<T>]) -> TaylorPoly<T> {
        let numerator = self.polynomial.eval(inputs);
        let mut denominator = TaylorPoly::one();
        for (v, geo_param) in self.geo_params.iter().enumerate() {
            denominator *= TaylorPoly::one()
                - TaylorPoly::from(T::from(geo_param.extract_constant().unwrap()))
                    * inputs[v].clone();
        }
        numerator / denominator
    }

    fn total_mass(&self) -> SymExpr {
        let numer = self
            .polynomial
            .eval_expr(&vec![1.0; self.polynomial.coeffs.ndim()]);
        let mut denom = SymExpr::one();
        for geo_param in &self.geo_params {
            denom *= SymExpr::one() - geo_param.clone();
        }
        numer / denom
    }
}

impl std::fmt::Display for GeometricBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.polynomial)?;
        writeln!(f, "______________________________________________________")?;
        for (i, param) in self.geo_params.iter().enumerate() {
            write!(f, "(1 - {param} * {})", Var(i))?;
        }
        writeln!(f)
    }
}

impl std::ops::Mul<SymExpr> for GeometricBound {
    type Output = Self;

    fn mul(mut self, rhs: SymExpr) -> Self::Output {
        self.polynomial *= rhs;
        self
    }
}

impl std::ops::Div<SymExpr> for GeometricBound {
    type Output = Self;

    fn div(mut self, rhs: SymExpr) -> Self::Output {
        self.polynomial /= rhs;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymExpr {
    Constant(f64),
    Variable(usize),
    Add(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    Pow(Box<SymExpr>, i32),
}

impl SymExpr {
    pub fn var(i: usize) -> Self {
        Self::Variable(i)
    }

    pub fn inverse(self) -> Self {
        self.pow(-1)
    }

    pub fn pow(self, n: i32) -> Self {
        if n == 0 {
            Self::one()
        } else if n == 1 || (n >= 0 && self.is_zero()) || self.is_one() {
            self
        } else {
            Self::Pow(Box::new(self), n)
        }
    }

    /// Must equal `rhs`.
    pub fn must_eq(self, rhs: Self) -> SymConstraint {
        SymConstraint::Eq(self, rhs)
    }

    /// Must be less than `rhs`.
    pub fn must_lt(self, rhs: Self) -> SymConstraint {
        SymConstraint::Lt(self, rhs)
    }

    /// Must be less than or equal to `rhs`.
    pub fn must_le(self, rhs: Self) -> SymConstraint {
        SymConstraint::Le(self, rhs)
    }

    /// Must be greater than `rhs`.
    pub fn must_gt(self, rhs: Self) -> SymConstraint {
        SymConstraint::Lt(rhs, self)
    }

    /// Must be greater than or equal to `rhs`.
    pub fn must_ge(self, rhs: Self) -> SymConstraint {
        SymConstraint::Le(rhs, self)
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> Self {
        match self {
            SymExpr::Constant(_) => self.clone(),
            SymExpr::Variable(i) => replacements[*i].clone(),
            SymExpr::Add(lhs, rhs) => lhs.substitute(replacements) + rhs.substitute(replacements),
            SymExpr::Mul(lhs, rhs) => lhs.substitute(replacements) * rhs.substitute(replacements),
            SymExpr::Pow(base, n) => base.substitute(replacements).pow(*n),
        }
    }

    pub fn extract_constant(&self) -> Option<f64> {
        match self {
            SymExpr::Constant(c) => Some(*c),
            _ => None,
        }
    }

    pub fn extract_linear(&self) -> Option<LinearExpr> {
        match self {
            SymExpr::Constant(c) => Some(LinearExpr::constant(*c)),
            SymExpr::Variable(i) => Some(LinearExpr::var(*i)),
            SymExpr::Add(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                Some(lhs + rhs)
            }
            SymExpr::Mul(lhs, rhs) => {
                let lhs = lhs.extract_linear()?;
                let rhs = rhs.extract_linear()?;
                if let Some(factor) = lhs.as_constant() {
                    Some(rhs * factor)
                } else if let Some(factor) = rhs.as_constant() {
                    Some(lhs * factor)
                } else {
                    None
                }
            }
            SymExpr::Pow(base, n) => {
                if *n == 0 {
                    return Some(LinearExpr::constant(1.0));
                }
                let base = base.extract_linear()?;
                if let Some(base) = base.as_constant() {
                    return Some(LinearExpr::constant(base.powi(*n)));
                }
                if *n == 1 {
                    Some(base)
                } else {
                    None
                }
            }
        }
    }

    pub fn to_z3<'a>(&self, ctx: &'a z3::Context) -> z3::ast::Real<'a> {
        match self {
            SymExpr::Constant(f) => {
                if !f.is_finite() {
                    unreachable!("Non-finite f64 in constraint: {f}");
                }
                let bits: u64 = f.to_bits();
                let sign: i64 = if bits >> 63 == 0 { 1 } else { -1 };
                let mut exponent = ((bits >> 52) & 0x7ff) as i64;
                let mantissa = if exponent == 0 {
                    (bits & 0x000f_ffff_ffff_ffff) << 1
                } else {
                    (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
                } as i64;
                // Exponent bias + mantissa shift
                exponent -= 1023 + 52;
                let m = z3::ast::Int::from_i64(ctx, sign * mantissa).to_real();
                let two = z3::ast::Int::from_i64(ctx, 2).to_real();
                let e = z3::ast::Int::from_i64(ctx, exponent).to_real();
                m * two.power(&e)
            }
            SymExpr::Variable(v) => z3::ast::Real::new_const(ctx, *v as u32),
            SymExpr::Add(e1, e2) => e1.to_z3(ctx) + e2.to_z3(ctx),
            SymExpr::Mul(e1, e2) => e1.to_z3(ctx) * e2.to_z3(ctx),
            SymExpr::Pow(e, n) => e
                .to_z3(ctx)
                .power(&z3::ast::Int::from_i64(ctx, (*n).into()).to_real()),
        }
    }

    fn to_python(&self) -> String {
        match self {
            SymExpr::Constant(c) => c.to_string(),
            SymExpr::Variable(v) => format!("x[{v}]"),
            SymExpr::Add(lhs, rhs) => format!("({} + {})", lhs.to_python(), rhs.to_python()),
            SymExpr::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python(), rhs.to_python()),
            SymExpr::Pow(lhs, rhs) => format!("({} ** {})", lhs.to_python(), rhs),
        }
    }

    fn to_python_z3(&self) -> String {
        match self {
            SymExpr::Constant(c) => c.to_string(),
            SymExpr::Variable(v) => format!("x{v}"),
            SymExpr::Add(lhs, rhs) => format!("({} + {})", lhs.to_python_z3(), rhs.to_python_z3()),
            SymExpr::Mul(lhs, rhs) => format!("({} * {})", lhs.to_python_z3(), rhs.to_python_z3()),
            SymExpr::Pow(lhs, rhs) => format!("({} ^ {})", lhs.to_python_z3(), rhs),
        }
    }

    fn eval(&self, values: &[f64]) -> f64 {
        match self {
            SymExpr::Constant(c) => *c,
            SymExpr::Variable(v) => values[*v],
            SymExpr::Add(lhs, rhs) => lhs.eval(values) + rhs.eval(values),
            SymExpr::Mul(lhs, rhs) => lhs.eval(values) * rhs.eval(values),
            SymExpr::Pow(base, n) => base.eval(values).powi(*n),
        }
    }
}

impl From<f64> for SymExpr {
    fn from(value: f64) -> Self {
        Self::Constant(value)
    }
}

impl Zero for SymExpr {
    fn zero() -> Self {
        Self::Constant(0.0)
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_zero(),
            _ => false,
        }
    }
}

impl One for SymExpr {
    fn one() -> Self {
        Self::Constant(1.0)
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Constant(x) => x.is_one(),
            _ => false,
        }
    }
}

impl std::ops::Neg for SymExpr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else if let Self::Constant(c) = self {
            (-c).into()
        } else {
            self * (-1.0).into()
        }
    }
}

impl std::ops::Add for SymExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            Self::Add(Box::new(self), Box::new(rhs))
        }
    }
}

impl std::ops::AddAssign for SymExpr {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Sub for SymExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self == rhs {
            Self::zero()
        } else {
            self + (-rhs)
        }
    }
}

impl std::ops::SubAssign for SymExpr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl std::ops::Mul for SymExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else if self.is_one() {
            rhs
        } else if rhs.is_one() {
            self
        } else {
            Self::Mul(Box::new(self), Box::new(rhs))
        }
    }
}

impl std::ops::MulAssign for SymExpr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl std::ops::Div for SymExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1)
    }
}

impl std::fmt::Display for SymExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => {
                if value < &0.0 {
                    write!(f, "(- {})", -value)
                } else {
                    write!(f, "{value}")
                }
            }
            Self::Variable(i) => write!(f, "x{i}"),
            Self::Add(lhs, rhs) => write!(f, "(+ {lhs} {rhs})"),
            Self::Mul(lhs, rhs) => {
                if Self::Constant(-1.0) == **rhs {
                    write!(f, "(- {lhs})")
                } else {
                    write!(f, "(* {lhs} {rhs})")
                }
            }
            Self::Pow(expr, n) => {
                if *n == -1 {
                    write!(f, "(/ 1 {expr})")
                } else if *n < 0 {
                    write!(f, "(/ 1 (^ {expr} {}))", -n)
                } else {
                    write!(f, "(^ {expr} {n})")
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum SymConstraint {
    Eq(SymExpr, SymExpr),
    Lt(SymExpr, SymExpr),
    Le(SymExpr, SymExpr),
    Or(Vec<SymConstraint>),
}
impl SymConstraint {
    pub fn or(constraints: Vec<SymConstraint>) -> Self {
        Self::Or(constraints)
    }

    fn to_z3<'a>(&self, ctx: &'a z3::Context) -> z3::ast::Bool<'a> {
        match self {
            SymConstraint::Eq(e1, e2) => z3::ast::Ast::_eq(&e1.to_z3(ctx), &e2.to_z3(ctx)),
            SymConstraint::Lt(e1, e2) => e1.to_z3(ctx).lt(&e2.to_z3(ctx)),
            SymConstraint::Le(e1, e2) => e1.to_z3(ctx).le(&e2.to_z3(ctx)),
            SymConstraint::Or(constraints) => {
                let disjuncts = constraints.iter().map(|c| c.to_z3(ctx)).collect::<Vec<_>>();
                z3::ast::Bool::or(ctx, &disjuncts.iter().collect::<Vec<_>>())
            }
        }
    }

    fn to_python_z3(&self) -> String {
        match self {
            SymConstraint::Eq(lhs, rhs) => {
                format!("{} == {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymConstraint::Lt(lhs, rhs) => {
                format!("{} < {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymConstraint::Le(lhs, rhs) => {
                format!("{} <= {}", lhs.to_python_z3(), rhs.to_python_z3())
            }
            SymConstraint::Or(cs) => {
                let mut res = "Or(".to_owned();
                let mut first = true;
                for c in cs {
                    if first {
                        first = false;
                    } else {
                        res += ", ";
                    }
                    res += &c.to_python_z3();
                }
                res + ")"
            }
        }
    }

    pub fn substitute(&self, replacements: &[SymExpr]) -> SymConstraint {
        match self {
            SymConstraint::Eq(e1, e2) => {
                SymConstraint::Eq(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Lt(e1, e2) => {
                SymConstraint::Lt(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Le(e1, e2) => {
                SymConstraint::Le(e1.substitute(replacements), e2.substitute(replacements))
            }
            SymConstraint::Or(constraints) => SymConstraint::Or(
                constraints
                    .iter()
                    .map(|c| c.substitute(replacements))
                    .collect(),
            ),
        }
    }

    fn extract_linear(&self) -> Option<LinearConstraint> {
        match self {
            SymConstraint::Eq(e1, e2) => Some(LinearConstraint::eq(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Lt(..) => None,
            SymConstraint::Le(e1, e2) => Some(LinearConstraint::le(
                e1.extract_linear()?,
                e2.extract_linear()?,
            )),
            SymConstraint::Or(constraints) => {
                // Here we only support constraints without variables
                for constraint in constraints {
                    if let Some(linear_constraint) = constraint.extract_linear() {
                        if linear_constraint.eval_constant() == Some(true) {
                            return Some(LinearConstraint::eq(
                                LinearExpr::constant(0.0),
                                LinearExpr::constant(0.0),
                            ));
                        }
                    }
                }
                return None;
            }
        }
    }
}

impl std::fmt::Display for SymConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq(e1, e2) => write!(f, "(= {e1} {e2})"),
            Self::Lt(e1, e2) => write!(f, "(< {e1} {e2})"),
            Self::Le(e1, e2) => write!(f, "(<= {e1} {e2})"),
            Self::Or(constraints) => {
                write!(f, "(or")?;
                for constraint in constraints {
                    write!(f, " {}", constraint)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SymPolynomial {
    coeffs: ArrayD<SymExpr>,
}

impl SymPolynomial {
    #[inline]
    pub fn new(coeffs: ArrayD<SymExpr>) -> Self {
        Self { coeffs }
    }

    pub fn var(var: Var) -> Self {
        Self::var_power(var, 1)
    }

    pub fn var_power(Var(v): Var, n: u32) -> SymPolynomial {
        if n == 0 {
            return Self::one();
        }
        let mut shape = vec![1; v + 1];
        shape[v] = n as usize + 1;
        let mut coeffs = ArrayD::zeros(shape);
        *coeffs
            .index_axis_mut(Axis(v), n as usize)
            .first_mut()
            .unwrap() = SymExpr::one();
        Self { coeffs }
    }

    #[inline]
    fn max_shape(&self, other: &Self) -> Vec<usize> {
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for (v, dim) in shape.iter_mut().enumerate() {
            if v < self.coeffs.ndim() {
                *dim = (*dim).max(self.coeffs.len_of(Axis(v)));
            }
            if v < other.coeffs.ndim() {
                *dim = (*dim).max(other.coeffs.len_of(Axis(v)));
            }
        }
        shape
    }

    #[inline]
    fn sum_shape(&self, other: &Self) -> Vec<usize> {
        let mut shape = vec![1; self.coeffs.ndim().max(other.coeffs.ndim())];
        for (v, dim) in shape.iter_mut().enumerate() {
            if v < self.coeffs.ndim() {
                *dim += self.coeffs.len_of(Axis(v)) - 1;
            }
            if v < other.coeffs.ndim() {
                *dim += other.coeffs.len_of(Axis(v)) - 1;
            }
        }
        shape
    }

    fn marginalize(&self, Var(v): Var) -> SymPolynomial {
        let mut result_shape = self.coeffs.shape().to_vec();
        if v >= result_shape.len() {
            return self.clone();
        }
        result_shape[v] = 1;
        let mut result = ArrayD::zeros(result_shape);
        for coeff in self.coeffs.axis_chunks_iter(Axis(v), 1) {
            result += &coeff;
        }
        Self::new(result)
    }

    fn coeff_of_var_power(&self, Var(v): Var, order: usize) -> SymPolynomial {
        if v >= self.coeffs.ndim() {
            if order == 0 {
                return self.clone();
            }
            return Self::zero();
        }
        if order >= self.coeffs.len_of(Axis(v)) {
            return Self::zero();
        }
        Self::new(
            self.coeffs
                .slice_axis(Axis(v), Slice::from(order..=order))
                .to_owned(),
        )
    }

    fn shift_coeffs_left(&self, Var(v): Var) -> SymPolynomial {
        if v >= self.coeffs.ndim() {
            return SymPolynomial::zero();
        }
        Self::new(self.coeffs.slice_axis(Axis(v), Slice::from(1..)).to_owned())
    }

    fn extract_zero_and_shift_left(self, var: Var) -> (SymPolynomial, SymPolynomial) {
        let p0 = self.coeff_of_var_power(var, 0);
        let rest = self - p0.clone();
        let shifted = rest.shift_coeffs_left(var);
        (p0, shifted)
    }

    fn substitute(&self, replacements: &[SymExpr]) -> SymPolynomial {
        Self::new(self.coeffs.map(|c| c.substitute(replacements)))
    }

    fn eval<T: From<f64> + Number>(&self, inputs: &[TaylorPoly<T>]) -> TaylorPoly<T> {
        let coeffs = self
            .coeffs
            .map(|sym_expr| T::from(sym_expr.extract_constant().unwrap()))
            .to_owned();
        let mut taylor = TaylorPoly::new(coeffs, vec![usize::MAX; inputs.len()]);
        for (v, input) in inputs.iter().enumerate() {
            taylor = taylor.subst_var(Var(v), input);
        }
        taylor
    }

    fn eval_expr_impl(array: &ArrayViewD<SymExpr>, points: &[f64]) -> SymExpr {
        let nvars = array.ndim();
        if nvars == 0 {
            return array.first().unwrap().clone();
        }
        let mut res = SymExpr::zero();
        for c in array.axis_iter(Axis(nvars - 1)) {
            res *= SymExpr::from(points[nvars - 1]);
            res += SymPolynomial::eval_expr_impl(&c, points);
        }
        res
    }

    pub fn eval_expr(&self, points: &[f64]) -> SymExpr {
        SymPolynomial::eval_expr_impl(&self.coeffs.view(), points)
    }
}

impl From<f64> for SymPolynomial {
    #[inline]
    fn from(value: f64) -> Self {
        Self::new(arr0(value.into()).into_dyn())
    }
}

impl Zero for SymPolynomial {
    #[inline]
    fn zero() -> Self {
        Self::new(arr0(SymExpr::zero()).into_dyn())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(SymExpr::is_zero)
    }
}

impl One for SymPolynomial {
    #[inline]
    fn one() -> Self {
        Self::new(arr0(SymExpr::one()).into_dyn())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs.first().unwrap().is_one()
    }
}

impl std::ops::Mul<SymExpr> for SymPolynomial {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SymExpr) -> Self::Output {
        Self::new(self.coeffs.mapv(|c| c * rhs.clone()))
    }
}

impl std::ops::MulAssign<SymExpr> for SymPolynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: SymExpr) {
        self.coeffs.mapv_inplace(|c| c * rhs.clone());
    }
}

impl std::ops::Div<SymExpr> for SymPolynomial {
    type Output = Self;

    #[inline]
    fn div(self, rhs: SymExpr) -> Self::Output {
        Self::new(self.coeffs.mapv(|c| c / rhs.clone()))
    }
}

impl std::ops::DivAssign<SymExpr> for SymPolynomial {
    #[inline]
    fn div_assign(&mut self, rhs: SymExpr) {
        self.coeffs.mapv_inplace(|c| c / rhs.clone());
    }
}

impl std::ops::Neg for SymPolynomial {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.coeffs.mapv_inplace(|c| -c);
        self
    }
}

#[inline]
fn broadcast<T>(xs: &mut ArrayD<T>, ys: &mut ArrayD<T>) {
    if xs.ndim() < ys.ndim() {
        for i in xs.ndim()..ys.ndim() {
            xs.insert_axis_inplace(Axis(i));
        }
    }
    if ys.ndim() < xs.ndim() {
        for i in ys.ndim()..xs.ndim() {
            ys.insert_axis_inplace(Axis(i));
        }
    }
}

impl std::ops::Add for SymPolynomial {
    type Output = Self;

    #[inline]
    fn add(mut self, mut other: Self) -> Self::Output {
        let result_shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(result_shape);
        broadcast(&mut self.coeffs, &mut other.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .add_assign(&other.coeffs);
        Self::new(result)
    }
}

impl std::ops::AddAssign for SymPolynomial {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl std::ops::Sub for SymPolynomial {
    type Output = Self;

    #[inline]
    fn sub(mut self, mut other: Self) -> Self::Output {
        let result_shape = self.max_shape(&other);
        let mut result = ArrayD::zeros(result_shape);
        broadcast(&mut self.coeffs, &mut other.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..self.coeffs.len_of(ax.axis)))
            .assign(&self.coeffs);
        result
            .slice_each_axis_mut(|ax| Slice::from(0..other.coeffs.len_of(ax.axis)))
            .sub_assign(&other.coeffs);
        Self::new(result)
    }
}

impl std::ops::SubAssign for SymPolynomial {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl std::ops::Mul for SymPolynomial {
    type Output = Self;

    #[inline]
    fn mul(mut self, mut other: Self) -> Self::Output {
        fn mul_helper(
            xs: &ArrayViewD<SymExpr>,
            ys: &ArrayViewD<SymExpr>,
            res: &mut ArrayViewMutD<SymExpr>,
        ) {
            if res.is_empty() {
                return;
            }
            if res.ndim() == 0 {
                *res.first_mut().unwrap() +=
                    xs.first().unwrap().clone() * ys.first().unwrap().clone();
                return;
            }
            for (k, mut z) in res.axis_iter_mut(Axis(0)).enumerate() {
                let lo = (k + 1).saturating_sub(ys.len_of(Axis(0)));
                let hi = (k + 1).min(xs.len_of(Axis(0)));
                for j in lo..hi {
                    mul_helper(
                        &xs.index_axis(Axis(0), j),
                        &ys.index_axis(Axis(0), k - j),
                        &mut z,
                    );
                }
            }
        }

        // Recognize multiplication by zero:
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        // Broadcast to common shape:
        broadcast(&mut self.coeffs, &mut other.coeffs);
        let result_shape = self.sum_shape(&other);

        // Recognize multiplication by one:
        if self.is_one() {
            return other;
        }
        if other.is_one() {
            return self;
        }

        let mut result = ArrayD::zeros(result_shape);
        mul_helper(
            &self.coeffs.view(),
            &other.coeffs.view(),
            &mut result.view_mut(),
        );
        Self::new(result)
    }
}

impl std::ops::MulAssign for SymPolynomial {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl std::fmt::Display for SymPolynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.coeffs.shape();
        let mut first = true;
        let mut empty_output = true;
        for index in indices(shape) {
            if self.coeffs[&index].is_zero() {
                continue;
            }
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.coeffs[&index])?;
            empty_output = false;
            for (i, exponent) in index.as_array_view().into_iter().enumerate() {
                if *exponent == 0 {
                    continue;
                }
                write!(f, "{}", crate::ppl::Var(i))?;
                if *exponent > 1 {
                    write!(f, "^{exponent}")?;
                }
            }
        }
        if empty_output {
            write!(f, "0")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LinearExpr {
    pub coeffs: Vec<f64>,
    pub constant: f64,
}

impl LinearExpr {
    pub fn new(coeffs: Vec<f64>, constant: f64) -> Self {
        Self { coeffs, constant }
    }

    pub fn zero() -> Self {
        Self::new(vec![], 0.0)
    }

    pub fn one() -> Self {
        Self::new(vec![1.0], 0.0)
    }

    pub fn constant(constant: f64) -> Self {
        Self::new(vec![], constant)
    }

    pub fn var(var: usize) -> Self {
        let mut coeffs = vec![0.0; var + 1];
        coeffs[var] = 1.0;
        Self::new(coeffs, 0.0)
    }

    pub fn as_constant(&self) -> Option<f64> {
        if self.coeffs.iter().all(|c| c == &0.0) {
            Some(self.constant)
        } else {
            None
        }
    }

    pub fn to_lp_expr(&self, vars: &[good_lp::Variable]) -> good_lp::Expression {
        let mut result = good_lp::Expression::from(self.constant);
        for (coeff, var) in self.coeffs.iter().zip(vars) {
            result.add_mul(*coeff, var);
        }
        result
    }
}

impl std::fmt::Display for LinearExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate() {
            if *coeff == 0.0 {
                continue;
            }
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            if *coeff == 1.0 {
                write!(f, "{}", SymExpr::var(i))?;
            } else if *coeff == -1.0 {
                write!(f, "-{}", SymExpr::var(i))?;
            } else {
                write!(f, "{}{}", coeff, SymExpr::var(i))?;
            }
        }
        if self.constant != 0.0 {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.constant)?;
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl std::ops::Neg for LinearExpr {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self * (-1.0)
    }
}

impl std::ops::Add for LinearExpr {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self::Output {
        let constant = self.constant + other.constant;
        let (mut coeffs, other) = if self.coeffs.len() > other.coeffs.len() {
            (self.coeffs, other.coeffs)
        } else {
            (other.coeffs, self.coeffs)
        };
        for i in 0..other.len() {
            coeffs[i] += other[i];
        }
        Self::new(coeffs, constant)
    }
}

impl std::ops::Sub for LinearExpr {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl std::ops::Mul<f64> for LinearExpr {
    type Output = Self;

    #[inline]
    fn mul(self, other: f64) -> Self::Output {
        Self::new(
            self.coeffs.into_iter().map(|c| c * other).collect(),
            self.constant * other,
        )
    }
}

#[derive(Clone, Debug)]
struct LinearConstraint {
    expr: LinearExpr,
    /// If true, `expr` must be equal to zero, otherwise it must be non-positive
    eq_zero: bool,
}

impl LinearConstraint {
    fn eq(e1: LinearExpr, e2: LinearExpr) -> Self {
        Self {
            expr: e1 - e2,
            eq_zero: true,
        }
    }

    fn le(e1: LinearExpr, e2: LinearExpr) -> Self {
        Self {
            expr: e1 - e2,
            eq_zero: false,
        }
    }

    fn to_lp_constraint(&self, var_list: &[good_lp::Variable]) -> good_lp::Constraint {
        let result = self.expr.to_lp_expr(var_list);
        if self.eq_zero {
            result.eq(0.0)
        } else {
            result.leq(0.0)
        }
    }

    fn eval_constant(&self) -> Option<bool> {
        let constant = self.expr.as_constant()?;
        if self.eq_zero {
            Some(constant == 0.0)
        } else {
            Some(constant <= 0.0)
        }
    }
}

impl std::fmt::Display for LinearConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.eq_zero {
            write!(f, "{} = 0", self.expr)
        } else {
            write!(f, "{} <= 0", self.expr)
        }
    }
}
