use ndarray::{ArrayD, ArrayViewD, Axis, Dimension, Slice};
use num_traits::{One, Zero};

use crate::{
    bounds::{
        bound::{BoundResult, GeometricBound},
        sym_expr::{SymConstraint, SymExpr},
        util::rational_to_qepcad,
    },
    number::{Number, Rational},
    ppl::{Distribution, Event, Natural, Program, Statement, Var},
    semantics::{
        support::{SupportTransformer, VarSupport},
        Transformer,
    },
    support::SupportSet,
};

pub struct BoundCtx {
    verbose: bool,
    default_unroll: usize,
    min_degree: usize,
    evt: bool,
    do_while_transform: bool,
    support: SupportTransformer,
    program_var_count: usize,
    sym_var_count: usize,
    // Variables used nonlinearly, in [0,1)
    nonlinear_vars: Vec<usize>,
    geom_vars: Vec<usize>,
    factor_vars: Vec<usize>,
    coeff_vars: Vec<usize>,
    constraints: Vec<SymConstraint>,
}

impl Default for BoundCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for BoundCtx {
    type Domain = BoundResult;

    fn init(&mut self, program: &Program) -> Self::Domain {
        self.program_var_count = program.used_vars().num_vars();
        let init_bound = GeometricBound {
            masses: ArrayD::ones(vec![1; self.program_var_count]),
            geo_params: vec![SymExpr::zero(); self.program_var_count],
        };
        BoundResult {
            bound: init_bound,
            var_supports: self.support.init(program),
        }
    }

    fn transform_event(
        &mut self,
        event: &Event,
        mut init: Self::Domain,
    ) -> (Self::Domain, Self::Domain) {
        match event {
            Event::InSet(v, set) => {
                let max = set.iter().fold(0, |acc, x| acc.max(x.0 as usize));
                init.bound.extend_axis(*v, max + 2);
                let axis = Axis(v.id());
                let len = init.bound.masses.len_of(axis);
                let mut then_bound = init.bound.clone();
                let mut else_bound = init.bound;
                then_bound.geo_params[v.id()] = SymExpr::zero();
                for i in 0..len {
                    if set.contains(&Natural(i as u32)) {
                        else_bound
                            .masses
                            .index_axis_mut(axis, i)
                            .fill(SymExpr::zero());
                    } else {
                        then_bound
                            .masses
                            .index_axis_mut(axis, i)
                            .fill(SymExpr::zero());
                    }
                }
                let (then_support, else_support) =
                    self.support.transform_event(event, init.var_supports);
                let then_res = BoundResult {
                    bound: then_bound,
                    var_supports: then_support,
                };
                let else_res = BoundResult {
                    bound: else_bound,
                    var_supports: else_support,
                };
                (then_res, else_res)
            }
            Event::DataFromDist(data, dist) => {
                if let Distribution::Bernoulli(p) = dist {
                    let p = Rational::from_ratio(p.numer, p.denom);
                    let p_compl = Rational::one() - p.clone();
                    let (then, els) = match data.0 {
                        0 => (p_compl, p),
                        1 => (p, p_compl),
                        _ => (Rational::zero(), Rational::one()),
                    };
                    let mut then_res = init.clone();
                    let mut else_res = init;
                    then_res.bound *= SymExpr::from(then);
                    else_res.bound *= SymExpr::from(els);
                    (then_res, else_res)
                } else {
                    todo!()
                }
            }
            Event::VarComparison(..) => todo!(),
            Event::Complement(event) => {
                let (then_res, else_res) = self.transform_event(event, init);
                (else_res, then_res)
            }
            Event::Intersection(events) => {
                let mut else_res = BoundResult {
                    bound: GeometricBound::zero(init.bound.geo_params.len()),
                    var_supports: VarSupport::Empty(init.var_supports.num_vars()),
                };
                let mut then_res = init;
                for event in events {
                    let (new_then, new_else) = self.transform_event(event, then_res);
                    then_res = new_then;
                    else_res = self.add_bound_results(else_res, new_else);
                }
                (then_res, else_res)
            }
        }
    }

    fn transform_statement(&mut self, stmt: &Statement, init: Self::Domain) -> Self::Domain {
        let direct_var_supports = if cfg!(debug_assertions) {
            Some(
                self.support
                    .transform_statement(stmt, init.var_supports.clone()),
            )
        } else {
            None
        };
        let result = match stmt {
            Statement::Sample {
                var,
                distribution,
                add_previous_value,
            } => {
                let new_var_info = SupportTransformer::transform_distribution(
                    distribution,
                    *var,
                    init.var_supports.clone(),
                    *add_previous_value,
                );
                let mut res = if *add_previous_value {
                    init
                } else {
                    init.marginalize(*var)
                };
                match distribution {
                    Distribution::Bernoulli(p) => {
                        let p = Rational::from_ratio(p.numer, p.denom);
                        res.bound
                            .add_categorical(*var, &[Rational::one() - p.clone(), p]);
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        let p = Rational::from_ratio(p.numer, p.denom);
                        res.bound *= SymExpr::from(p.clone());
                        res.bound.geo_params[var.id()] = SymExpr::from(Rational::one() - p);
                    }
                    Distribution::Uniform { start, end } => {
                        let p = Rational::one() / Rational::from_int(end.0 - start.0);
                        let categorical = (0..end.0)
                            .map(|x| {
                                if x < start.0 {
                                    Rational::zero()
                                } else {
                                    p.clone()
                                }
                            })
                            .collect::<Vec<_>>();
                        res.bound.add_categorical(*var, &categorical);
                    }
                    Distribution::Categorical(categorical) => {
                        res.bound.add_categorical(
                            *var,
                            &categorical
                                .iter()
                                .map(|p| Rational::from_ratio(p.numer, p.denom))
                                .collect::<Vec<_>>(),
                        );
                    }
                    _ => todo!(),
                };
                res.var_supports = new_var_info;
                res
            }
            Statement::Assign {
                var,
                add_previous_value,
                addend,
                offset,
            } => {
                let mut new_bound = if *add_previous_value {
                    init
                } else {
                    init.marginalize(*var)
                };
                match (addend, offset) {
                    (Some((Natural(1), w)), Natural(0)) => {
                        if let Some(range) = new_bound.var_supports[w].finite_nonempty_range() {
                            let mut new_shape = new_bound.bound.masses.shape().to_vec();
                            new_shape[var.id()] += *range.end() as usize;
                            let new_len = new_shape[var.id()];
                            new_bound.bound.extend_axis(*var, new_len);
                            let masses = &new_bound.bound.masses;
                            let mut res_masses = ArrayD::zeros(new_shape);
                            for i in range {
                                let i = i as usize;
                                res_masses
                                    .slice_axis_mut(Axis(w.id()), Slice::from(i..=i))
                                    .slice_axis_mut(Axis(var.id()), Slice::from(i..new_len))
                                    .assign(
                                        &masses
                                            .slice_axis(Axis(w.id()), Slice::from(i..=i))
                                            .slice_axis(
                                                Axis(var.id()),
                                                Slice::from(0..new_len - i),
                                            ),
                                    );
                            }
                            new_bound.bound.masses = res_masses;
                        } else {
                            todo!("Addition of a variable is not implemented for infinite support: {}", stmt.to_string());
                        }
                    }
                    (None, offset) => {
                        new_bound.bound.shift_right(*var, offset.0 as usize);
                    }
                    _ => todo!("{}", stmt.to_string()),
                }
                new_bound.var_supports = self
                    .support
                    .transform_statement(stmt, new_bound.var_supports);
                new_bound
            }
            Statement::Decrement { var, offset } => {
                let mut new_bound = init;
                new_bound.bound.shift_left(*var, offset.0 as usize);
                new_bound.var_supports = self
                    .support
                    .transform_statement(stmt, new_bound.var_supports);
                new_bound
            }
            Statement::IfThenElse { cond, then, els } => {
                let (then_res, else_res) = self.transform_event(cond, init);
                let then_res = self.transform_statements(then, then_res);
                let else_res = self.transform_statements(els, else_res);
                self.add_bound_results(then_res, else_res)
            }
            Statement::While { cond, unroll, body } => self.bound_while(cond, *unroll, body, init),
            Statement::Fail => BoundResult {
                bound: GeometricBound::zero(self.program_var_count),
                var_supports: VarSupport::empty(init.var_supports.num_vars()),
            },
            Statement::Normalize { .. } => todo!(),
        };
        if let Some(direct_var_info) = direct_var_supports {
            debug_assert_eq!(
                result.var_supports, direct_var_info,
                "inconsistent variable support info for:\n{stmt}"
            );
        }
        result
    }
}

impl BoundCtx {
    pub fn new() -> Self {
        Self {
            verbose: false,
            default_unroll: 8,
            min_degree: 1,
            evt: false,
            do_while_transform: false,
            support: SupportTransformer,
            program_var_count: 0,
            sym_var_count: 0,
            nonlinear_vars: Vec::new(),
            geom_vars: Vec::new(),
            factor_vars: Vec::new(),
            coeff_vars: Vec::new(),
            constraints: Vec::new(),
        }
    }

    pub fn with_min_degree(self, min_degree: usize) -> Self {
        Self { min_degree, ..self }
    }

    pub fn with_default_unroll(self, default_unroll: usize) -> Self {
        Self {
            default_unroll,
            ..self
        }
    }

    pub fn with_evt(self, evt: bool) -> Self {
        Self { evt, ..self }
    }

    pub fn with_do_while_transform(self, do_while_transform: bool) -> Self {
        Self {
            do_while_transform,
            ..self
        }
    }

    pub fn with_verbose(self, verbose: bool) -> Self {
        Self { verbose, ..self }
    }

    pub fn sym_var_count(&self) -> usize {
        self.sym_var_count
    }

    pub fn nonlinear_vars(&self) -> &[usize] {
        &self.nonlinear_vars
    }

    pub fn geom_vars(&self) -> &[usize] {
        &self.geom_vars
    }

    pub fn factor_vars(&self) -> &[usize] {
        &self.factor_vars
    }

    pub fn coefficient_vars(&self) -> &[usize] {
        &self.coeff_vars
    }

    pub fn constraints(&self) -> &[SymConstraint] {
        &self.constraints
    }

    fn fresh_sym_var_idx(&mut self) -> usize {
        let var = self.sym_var_count;
        self.sym_var_count += 1;
        var
    }

    pub fn add_constraint(&mut self, constraint: SymConstraint) {
        if !constraint.is_trivial() {
            self.constraints.push(constraint);
        }
    }

    pub fn new_geom_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx();
        let var = SymExpr::var(idx);
        self.nonlinear_vars.push(idx);
        self.geom_vars.push(idx);
        self.add_constraint(var.clone().must_ge(SymExpr::zero()));
        self.add_constraint(var.clone().must_lt(SymExpr::one()));
        var
    }

    pub fn new_factor_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx();
        let var = SymExpr::var(idx);
        self.nonlinear_vars.push(idx);
        self.factor_vars.push(idx);
        self.add_constraint(var.clone().must_ge(SymExpr::zero()));
        self.add_constraint(var.clone().must_lt(SymExpr::one()));
        var
    }

    pub fn new_coeff_var(&mut self) -> SymExpr {
        let idx = self.fresh_sym_var_idx();
        let var = SymExpr::var(idx);
        self.coeff_vars.push(idx);
        var
    }

    pub fn add_bounds(
        &mut self,
        mut lhs: GeometricBound,
        mut rhs: GeometricBound,
    ) -> GeometricBound {
        let count = self.program_var_count;
        let mut geo_params = Vec::with_capacity(count);
        for i in 0..count {
            let a = lhs.geo_params[i].clone();
            let b = rhs.geo_params[i].clone();
            if a == b {
                geo_params.push(a);
            } else if a.is_zero() {
                geo_params.push(b);
            } else if b.is_zero() {
                geo_params.push(a);
            } else {
                let new_geo_param_var = self.new_geom_var();
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
        for i in 0..count {
            let lhs_len = lhs.masses.len_of(Axis(i));
            let rhs_len = rhs.masses.len_of(Axis(i));
            if lhs_len < rhs_len {
                lhs.extend_axis(Var(i), rhs_len);
            } else if rhs_len < lhs_len {
                rhs.extend_axis(Var(i), lhs_len);
            }
        }
        GeometricBound {
            masses: lhs.masses + rhs.masses,
            geo_params,
        }
    }

    pub fn add_bound_results(&mut self, lhs: BoundResult, rhs: BoundResult) -> BoundResult {
        let bound = self.add_bounds(lhs.bound, rhs.bound);
        let var_supports = lhs.var_supports.join(&rhs.var_supports);
        BoundResult {
            bound,
            var_supports,
        }
    }

    fn bound_while(
        &mut self,
        cond: &Event,
        unroll: Option<usize>,
        body: &[Statement],
        init: BoundResult,
    ) -> BoundResult {
        let mut pre_loop = init;
        let mut rest = BoundResult {
            bound: GeometricBound::zero(self.program_var_count),
            var_supports: VarSupport::empty(pre_loop.var_supports.num_vars()),
        };
        let unroll_count = unroll.unwrap_or(self.default_unroll);
        let unroll_result =
            self.support
                .find_unroll_fixpoint(cond, body, pre_loop.var_supports.clone());
        let unroll_count = if let Some((iters, _, _)) = unroll_result {
            unroll_count.max(iters)
        } else {
            unroll_count
        };
        if self.verbose {
            println!("Unrolling {unroll_count} times");
        }
        for _ in 0..unroll_count {
            let (then_bound, else_bound) = self.transform_event(cond, pre_loop.clone());
            pre_loop = self.transform_statements(body, then_bound);
            rest = self.add_bound_results(rest, else_bound);
        }
        let invariant_supports =
            self.support
                .find_while_invariant(cond, body, pre_loop.var_supports.clone());
        let invariant_supports = if self.do_while_transform {
            let (invariant_entry, _) = self.support.transform_event(cond, invariant_supports);
            invariant_entry
        } else {
            invariant_supports
        };
        let shape = match &invariant_supports {
            VarSupport::Empty(num_vars) => vec![(0, Some(0)); *num_vars],
            VarSupport::Prod(supports) => supports
                .iter()
                .map(|s| match s {
                    SupportSet::Empty => (0, Some(0)),
                    SupportSet::Range { start, end } => {
                        (*start as usize, end.map(|x| x as usize + 1))
                    }
                    SupportSet::Interval { .. } => todo!(),
                })
                .collect::<Vec<_>>(),
        };
        if self.evt {
            let invariant = BoundResult {
                bound: self.new_bound(shape, self.min_degree),
                var_supports: invariant_supports,
            };
            if self.verbose {
                println!("EVT-invariant: {invariant}");
            }
            let (loop_entry, loop_exit) = self.transform_event(cond, invariant.clone());
            let one_iter = self.transform_statements(body, loop_entry);
            let rhs = self.add_bound_results(pre_loop, one_iter);
            if self.verbose {
                println!("Post loop body: {rhs}");
            }
            self.assert_le(&rhs.bound, &invariant.bound);
            self.add_bound_results(loop_exit, rest)
        } else {
            let invariant = BoundResult {
                bound: self.new_bound(shape, self.min_degree),
                var_supports: invariant_supports,
            };
            if self.verbose {
                println!("Invariant: {invariant}");
            }
            let (post_loop, mut exit) = if self.do_while_transform {
                let (loop_entry, loop_exit) = self.transform_event(cond, pre_loop);
                rest = self.add_bound_results(rest, loop_exit);
                self.assert_le(&loop_entry.bound, &invariant.bound);
                let post_loop = self.transform_statements(body, invariant.clone());
                self.transform_event(cond, post_loop)
            } else {
                self.assert_le(&pre_loop.bound, &invariant.bound);
                let (loop_entry, loop_exit) = self.transform_event(cond, invariant.clone());
                let post_loop = self.transform_statements(body, loop_entry);
                (post_loop, loop_exit)
            };
            if self.verbose {
                println!("Post loop body: {post_loop}");
            }
            let c = self.new_factor_var();
            if self.verbose {
                println!("Invariant-c: {c}");
            }
            self.add_constraint(c.clone().must_ge(SymExpr::zero()));
            self.add_constraint(c.clone().must_lt(SymExpr::one()));
            self.assert_le(&post_loop.bound, &(invariant.bound.clone() * c.clone()));
            exit.bound /= SymExpr::one() - c.clone();
            self.add_bound_results(exit, rest)
        }
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
            if self.nonlinear_vars.contains(&i) {
                write!(out, "0, ").unwrap();
            } else {
                write!(out, "-np.inf, ").unwrap();
            }
        }
        writeln!(out, "],").unwrap();
        write!(out, "  [").unwrap();
        for i in 0..self.sym_var_count {
            if self.nonlinear_vars.contains(&i) {
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
                SymConstraint::Eq(_, _) | SymConstraint::Or(_) => continue,
                SymConstraint::Lt(lhs, rhs) | SymConstraint::Le(lhs, rhs) => {
                    writeln!(out, "    ({}) - ({}),", rhs.to_python(), lhs.to_python()).unwrap();
                }
            }
        }
        writeln!(out, "  ]").unwrap();
        writeln!(out, "constraints = NonlinearConstraint(constraint_fun, 0, np.inf, jac='2-point', hess=BFGS())").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "def objective(x):").unwrap();
        let total = bound.total_mass();
        write!(out, "  return ({})", total.to_python()).unwrap();
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
            writeln!(out, "(assert {constraint})")?;
        }
        writeln!(out)?;
        writeln!(out, "(check-sat)")?;
        writeln!(out, "(get-model)")?;
        Ok(())
    }

    pub fn output_qepcad<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        // Name:
        writeln!(out, "[Constraints]")?;

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
            write!(out, "  {}", c.to_qepcad(&rational_to_qepcad))?;
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

    fn new_masses(&mut self, shape: Vec<(usize, usize)>) -> ArrayD<SymExpr> {
        let dims = shape.iter().map(|(_, e)| (*e).max(1)).collect::<Vec<_>>();
        let mut coeffs = ArrayD::zeros(dims);
        for (idx, c) in coeffs.indexed_iter_mut() {
            let is_nonzero = idx
                .as_array_view()
                .iter()
                .zip(&shape)
                .all(|(i, (start, end))| i >= start || (start >= end && i + 1 == *end));
            if is_nonzero {
                *c = self.new_coeff_var();
            }
        }
        coeffs
    }

    fn new_bound(
        &mut self,
        shape: Vec<(usize, Option<usize>)>,
        min_degree: usize,
    ) -> GeometricBound {
        let mut geo_params = vec![SymExpr::zero(); shape.len()];
        for (v, p) in geo_params.iter_mut().enumerate() {
            if shape[v].1.is_none() {
                *p = self.new_geom_var();
            }
        }
        let shape = shape
            .into_iter()
            .map(|(start, end)| (start, end.unwrap_or(min_degree)))
            .collect::<Vec<_>>();
        GeometricBound {
            masses: self.new_masses(shape),
            geo_params,
        }
    }

    fn assert_le_helper(
        &mut self,
        lhs_coeffs: &ArrayViewD<SymExpr>,
        lhs_factor: SymExpr,
        lhs_geo_params: &[SymExpr],
        rhs_coeffs: &ArrayViewD<SymExpr>,
        rhs_factor: SymExpr,
        rhs_geo_params: &[SymExpr],
    ) {
        if lhs_geo_params.is_empty() {
            assert!(lhs_coeffs.len() == 1);
            assert!(rhs_coeffs.len() == 1);
            let lhs_coeff = lhs_coeffs.first().unwrap().clone() * lhs_factor;
            let rhs_coeff = rhs_coeffs.first().unwrap().clone() * rhs_factor;
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
        for i in 0..len {
            let (lhs, lhs_factor) = if i < len_lhs {
                (lhs_coeffs.index_axis(Axis(0), i), lhs_factor.clone())
            } else {
                (
                    lhs_coeffs.index_axis(Axis(0), len_lhs - 1),
                    lhs_factor.clone() * lhs_geo_params[0].clone().pow((i - len_lhs + 1) as i32),
                )
            };
            let (rhs, rhs_factor) = if i < len_rhs {
                (rhs_coeffs.index_axis(Axis(0), i), rhs_factor.clone())
            } else {
                (
                    rhs_coeffs.index_axis(Axis(0), len_rhs - 1),
                    rhs_factor.clone() * rhs_geo_params[0].clone().pow((i - len_rhs + 1) as i32),
                )
            };
            self.assert_le_helper(
                &lhs,
                lhs_factor,
                &lhs_geo_params[1..],
                &rhs,
                rhs_factor,
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
            &lhs.masses.view(),
            SymExpr::one(),
            &lhs.geo_params,
            &rhs.masses.view(),
            SymExpr::one(),
            &rhs.geo_params,
        );
    }
}
