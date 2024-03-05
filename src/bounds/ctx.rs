use std::time::Duration;

use good_lp::{variable, ProblemVariables, Solution, SolverModel};
use ndarray::{ArrayD, Axis};
use num_traits::{One, Zero};

use crate::{
    bounds::{bound::*, sym_expr::*, sym_poly::*},
    number::{Number, F64},
    ppl::{Distribution, Event, Natural, Program, Statement},
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
                    Distribution::Geometric(_) => {
                        res[var.id()] += SupportSet::naturals();
                        res
                    }
                    Distribution::Uniform { start, end } => {
                        res[var.id()] += (start.0..end.0).into();
                        res
                    }
                    Distribution::Binomial(n, _) => {
                        res[var.id()] += (0..=n.0).into();
                        res
                    }
                    Distribution::BinomialVarTrials(w, _) => {
                        let w_support = res[w.id()].clone();
                        res[var.id()] += w_support;
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
        // TODO: upper bound of this for loop should be the highest constant occurring in the loop or something like that
        for i in 0..100 {
            let (new_loop_entry, new_loop_exit) =
                Self::one_iteration(loop_entry.clone(), loop_exit.clone(), body, cond);
            if loop_entry == new_loop_entry && loop_exit == new_loop_exit {
                return (Some(i), loop_entry, loop_exit);
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
                let min_degree = 1;
                let shape = invariant_supports
                    .iter()
                    .map(|s| match s {
                        SupportSet::Empty => 0,
                        SupportSet::Range { end, .. } => {
                            if let Some(end) = end {
                                *end as usize + 1
                            } else {
                                min_degree
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

    pub fn output_smt<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "(set-logic QF_NRA)")?;
        writeln!(out, "(set-option :pp.decimal true)")?;
        writeln!(out)?;
        for i in 0..self.sym_var_count() {
            writeln!(out, "(declare-const {} Real)", SymExpr::var(i))?;
        }
        writeln!(out)?;
        for constraint in &self.constraints {
            writeln!(out, "(assert {})", constraint)?;
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "(assert-soft {})", constraint)?;
        }
        writeln!(out)?;
        writeln!(out, "(check-sat)")?;
        writeln!(out, "(get-model)")?;
        Ok(())
    }

    pub fn output_qepcad<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        // Name:
        writeln!(out, "[Geometric bounds constraints]")?;

        // List of variables:
        write!(out, "(")?;
        let mut first = true;
        for i in 0..self.sym_var_count {
            if first {
                first = false;
            } else {
                write!(out, ", ")?;
            }
            write!(out, "{}", crate::ppl::Var(i))?;
        }
        writeln!(out, ")")?;

        // Number of free variables:
        writeln!(out, "2")?; // two variables for plotting

        // Formula:
        for i in 2..self.sym_var_count {
            writeln!(out, "(E {})", crate::ppl::Var(i))?;
        }
        writeln!(out, "[")?;
        let mut first = true;
        for c in self.constraints() {
            if first {
                first = false;
            } else {
                writeln!(out, r" /\")?;
            }
            write!(out, "  {}", c.to_qepcad())?;
        }
        for c in self.soft_constraints() {
            writeln!(out, r" /\")?;
            write!(out, "  {}", c.to_qepcad())?;
        }
        writeln!(out, "\n].")?;

        // Commands for various solving stages:
        writeln!(out, "go")?;
        writeln!(out, "go")?;
        writeln!(out, "go")?;
        writeln!(out, "p-2d-cad 0 1 0 1 0.0001 plot.eps")?; // 2D plot
        writeln!(out, "go")?;
        Ok(())
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
