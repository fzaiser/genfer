use std::time::Duration;

use good_lp::{variable, ProblemVariables, Solution, SolverModel};
use ndarray::{ArrayD, ArrayViewD, Axis, Slice};
use num_traits::{One, Zero};

use crate::{
    bounds::{
        bound::{BoundResult, GeometricBound},
        sym_expr::{SymConstraint, SymExpr},
        util::{f64_to_qepcad, f64_to_z3, z3_real_to_f64},
    },
    number::{Number, F64},
    ppl::{Distribution, Event, Natural, Program, Statement, Var},
    semantics::{
        support::{SupportTransformer, VarSupport},
        Transformer,
    },
    support::SupportSet,
};

#[derive(Clone, Debug)]
pub enum SolverError {
    Infeasible,
    Timeout,
}

pub struct BoundCtx {
    default_unroll: usize,
    min_degree: usize,
    support: SupportTransformer,
    program_var_count: usize,
    sym_var_count: usize,
    // Variables used nonlinearly, in [0,1)
    nonlinear_param_vars: Vec<usize>,
    constraints: Vec<SymConstraint<f64>>,
    soft_constraints: Vec<SymConstraint<f64>>,
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
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        let mut new_shape = res.bound.masses.shape().to_owned();
                        new_shape[var.id()] = 2;
                        res.bound.masses =
                            res.bound.masses.broadcast(new_shape).unwrap().to_owned();
                        res.bound
                            .masses
                            .index_axis_mut(Axis(var.id()), 0)
                            .map_inplace(|e| *e *= (1.0 - p).into());
                        res.bound
                            .masses
                            .index_axis_mut(Axis(var.id()), 1)
                            .map_inplace(|e| *e *= p.into());
                    }
                    Distribution::Geometric(p) if !add_previous_value => {
                        let p = F64::from_ratio(p.numer, p.denom).to_f64();
                        res.bound *= SymExpr::from(p);
                        res.bound.geo_params[var.id()] = SymExpr::from(1.0 - p);
                    }
                    Distribution::Uniform { start, end } => {
                        let mut new_shape = res.bound.masses.shape().to_owned();
                        let len = end.0 as usize;
                        new_shape[var.id()] = len;
                        res.bound.masses =
                            res.bound.masses.broadcast(new_shape).unwrap().to_owned();
                        for i in 0..len {
                            if i < start.0 as usize {
                                res.bound
                                    .masses
                                    .index_axis_mut(Axis(var.id()), i)
                                    .fill(SymExpr::zero());
                            } else {
                                res.bound
                                    .masses
                                    .index_axis_mut(Axis(var.id()), i)
                                    .map_inplace(|e| {
                                        *e /= SymExpr::from(f64::from(end.0 - start.0));
                                    });
                            }
                        }
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
                if let (None, Natural(n)) = (addend, offset) {
                    let mut new_bound = if *add_previous_value {
                        init
                    } else {
                        init.marginalize(*var)
                    };
                    let mut zero_shape = new_bound.bound.masses.shape().to_owned();
                    zero_shape[var.id()] = *n as usize;
                    new_bound.bound.masses = ndarray::concatenate(
                        Axis(var.id()),
                        &[
                            ArrayD::zeros(zero_shape).view(),
                            new_bound.bound.masses.view(),
                        ],
                    )
                    .unwrap();
                    new_bound.var_supports = self
                        .support
                        .transform_statement(stmt, new_bound.var_supports);
                    new_bound
                } else {
                    todo!()
                }
            }
            Statement::Decrement { var, offset } => {
                let mut cur = init;
                cur.bound.extend_axis(*var, (offset.0 + 2) as usize);
                let zero_elem = cur
                    .bound
                    .masses
                    .slice_axis(Axis(var.id()), Slice::from(0..=offset.0 as usize))
                    .sum_axis(Axis(var.id()));
                cur.bound
                    .masses
                    .slice_axis_inplace(Axis(var.id()), Slice::from(offset.0 as usize..));
                cur.bound
                    .masses
                    .index_axis_mut(Axis(var.id()), 0)
                    .assign(&zero_elem);
                cur.var_supports = self.support.transform_statement(stmt, cur.var_supports);
                cur
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
            default_unroll: 8,
            min_degree: 1,
            support: SupportTransformer,
            program_var_count: 0,
            sym_var_count: 0,
            nonlinear_param_vars: Vec::new(),
            constraints: Vec::new(),
            soft_constraints: Vec::new(),
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

    pub fn sym_var_count(&self) -> usize {
        self.sym_var_count
    }

    pub fn constraints(&self) -> &[SymConstraint<f64>] {
        &self.constraints
    }

    pub fn soft_constraints(&self) -> &[SymConstraint<f64>] {
        &self.soft_constraints
    }

    pub fn fresh_sym_var_idx(&mut self) -> usize {
        let var = self.sym_var_count;
        self.sym_var_count += 1;
        var
    }

    pub fn fresh_sym_var(&mut self) -> SymExpr<f64> {
        SymExpr::var(self.fresh_sym_var_idx())
    }

    pub fn add_constraint(&mut self, constraint: SymConstraint<f64>) {
        // println!("Adding constraint {constraint}");
        self.constraints.push(constraint);
    }

    pub fn add_soft_constraint(&mut self, constraint: SymConstraint<f64>) {
        // println!("Adding soft constraint {constraint}");
        self.soft_constraints.push(constraint);
    }

    pub fn new_geo_param_var(&mut self) -> SymExpr<f64> {
        let idx = self.fresh_sym_var_idx();
        let var = SymExpr::var(idx);
        self.nonlinear_param_vars.push(idx);
        self.add_constraint(var.clone().must_ge(SymExpr::zero()));
        self.add_constraint(var.clone().must_le(SymExpr::one()));
        self.add_soft_constraint(var.clone().must_lt(SymExpr::one()));
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
        let (iters, invariant_supports, _) =
            self.support
                .analyze_while(cond, body, pre_loop.var_supports.clone());
        let unroll_count = if let Some(iters) = iters {
            unroll_count.max(iters)
        } else {
            unroll_count
        };
        println!("Unrolling {unroll_count} times");
        for _ in 0..unroll_count {
            let (then_bound, else_bound) = self.transform_event(cond, pre_loop.clone());
            pre_loop = self.transform_statements(body, then_bound);
            rest = self.add_bound_results(rest, else_bound);
        }
        let shape = match &invariant_supports {
            VarSupport::Empty(num_vars) => vec![0; *num_vars],
            VarSupport::Prod(supports) => supports
                .iter()
                .map(|s| match s {
                    SupportSet::Empty => 0,
                    SupportSet::Range { end, .. } => {
                        if let Some(end) = end {
                            *end as usize + 1
                        } else {
                            self.min_degree
                        }
                    }
                    SupportSet::Interval { .. } => todo!(),
                })
                .collect::<Vec<_>>(),
        };
        let finite_supports = match &invariant_supports {
            VarSupport::Empty(num_vars) => vec![true; *num_vars],
            VarSupport::Prod(supports) => supports
                .iter()
                .map(|s| s.is_empty() || s.finite_nonempty_range().is_some())
                .collect::<Vec<_>>(),
        };
        let (_, invariant_supports, _) =
            self.support
                .analyze_while(cond, body, pre_loop.var_supports.clone());
        let (loop_entry, loop_exit) = self.transform_event(cond, pre_loop);
        rest = self.add_bound_results(rest, loop_exit);
        let invariant = BoundResult {
            bound: self.new_bound(shape, &finite_supports),
            var_supports: invariant_supports,
        };
        println!("Invariant: {invariant}");
        self.assert_le(&loop_entry.bound, &invariant.bound);
        let idx = self.fresh_sym_var_idx();
        let c = SymExpr::var(idx);
        println!("Invariant-c: {c}");
        self.nonlinear_param_vars.push(idx);
        self.add_constraint(c.clone().must_ge(SymExpr::zero()));
        self.add_constraint(c.clone().must_le(SymExpr::one()));
        self.add_soft_constraint(c.clone().must_lt(SymExpr::one()));
        let cur_bound = self.transform_statements(body, invariant.clone());
        let (post_loop, mut exit) = self.transform_event(cond, cur_bound);
        self.assert_le(&post_loop.bound, &(invariant.bound.clone() * c.clone()));
        exit.bound /= SymExpr::one() - c.clone();
        self.add_bound_results(exit, rest)
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
            writeln!(out, "(declare-const {} Real)", SymExpr::<f64>::var(i))?;
        }
        writeln!(out)?;
        for constraint in &self.constraints {
            writeln!(out, "(assert {constraint})")?;
        }
        for constraint in &self.soft_constraints {
            writeln!(out, "(assert-soft {constraint})")?;
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
            write!(out, "  {}", c.to_qepcad(&|f| f64_to_qepcad(*f)))?;
        }
        for c in self.soft_constraints() {
            writeln!(out, r" /\")?;
            write!(out, "  {}", c.to_qepcad(&|f| f64_to_qepcad(*f)))?;
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

    fn new_masses(&mut self, shape: Vec<usize>) -> ArrayD<SymExpr<f64>> {
        let mut coeffs = ArrayD::zeros(shape);
        for c in &mut coeffs {
            *c = self.fresh_sym_var();
        }
        coeffs
    }

    fn new_bound(&mut self, shape: Vec<usize>, finite_supports: &[bool]) -> GeometricBound {
        let mut geo_params = vec![SymExpr::zero(); shape.len()];
        for (v, p) in geo_params.iter_mut().enumerate() {
            if !finite_supports[v] {
                *p = self.new_geo_param_var();
            }
        }
        GeometricBound {
            masses: self.new_masses(shape),
            geo_params,
        }
    }

    fn assert_le_helper(
        &mut self,
        lhs_coeffs: &ArrayViewD<SymExpr<f64>>,
        lhs_factor: SymExpr<f64>,
        lhs_geo_params: &[SymExpr<f64>],
        rhs_coeffs: &ArrayViewD<SymExpr<f64>>,
        rhs_factor: SymExpr<f64>,
        rhs_geo_params: &[SymExpr<f64>],
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
        let total = bound.total_mass();
        let objective = total
            .extract_linear()
            .unwrap_or_else(|| panic!("Objective is not linear in the program variables: {total}"))
            .to_lp_expr(&var_list, &|v| *v);
        let mut problem = lp.minimise(objective).using(good_lp::default_solver);
        for constraint in &linear_constraints {
            problem.add_constraint(constraint.to_lp_constraint(&var_list, &|v| *v));
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
        for coeff in &mut resolved_bound.masses {
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
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(timeout.as_millis() as u64);
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);
        for constraint in self.constraints() {
            solver.assert(&constraint.to_z3(&ctx, &f64_to_z3));
        }
        for constraint in self.soft_constraints() {
            solver.assert(&constraint.to_z3(&ctx, &f64_to_z3));
        }
        solver.push();
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
        let mut best_obj_val = f64::INFINITY;
        let mut best_bound = bound.clone();
        let mut best_nonlinear_var_assignment = None::<Vec<f64>>;
        let mut obj_lo = 0.0;
        let mut obj_hi = f64::INFINITY;
        loop {
            if let Some(model) = solver.get_model() {
                for var in 0..self.sym_var_count() {
                    let val = model
                        .eval(&z3::ast::Real::new_const(&ctx, var as u32), false)
                        .unwrap();
                    let val = z3_real_to_f64(&val)
                        .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                    println!("{var} -> {val}", var = SymExpr::<f64>::var(var));
                }
                let mut resolved_bound = bound.clone();
                for coeff in &mut resolved_bound.masses {
                    let val = model.eval(&coeff.to_z3(&ctx, &f64_to_z3), false).unwrap();
                    let val = z3_real_to_f64(&val)
                        .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                    *coeff = SymExpr::Constant(val);
                }
                for geo_param in &mut resolved_bound.geo_params {
                    let val = model
                        .eval(&geo_param.to_z3(&ctx, &f64_to_z3), false)
                        .unwrap();
                    let val = z3_real_to_f64(&val)
                        .unwrap_or_else(|| panic!("{val} cannot be converted to f64"));
                    *geo_param = SymExpr::Constant(val);
                }
                println!("SMT solution:\n {resolved_bound}");
                let obj_val = z3_real_to_f64(
                    &model
                        .eval(&objective.to_z3(&ctx, &f64_to_z3), false)
                        .unwrap(),
                )
                .unwrap();
                println!("Total mass (objective): {obj_val}");
                if obj_val < best_obj_val {
                    best_obj_val = obj_val;
                    best_bound = resolved_bound;
                    best_nonlinear_var_assignment = Some(
                        self.nonlinear_param_vars
                            .iter()
                            .map(|var| {
                                let val = model
                                    .eval(&SymExpr::var(*var).to_z3(&ctx, &f64_to_z3), false)
                                    .unwrap();
                                z3_real_to_f64(&val)
                                    .unwrap_or_else(|| panic!("{val} cannot be converted to f64"))
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                obj_hi = best_obj_val;
            }
            println!("Objective bound: [{obj_lo}, {obj_hi}]");
            if !optimize || obj_hi - obj_lo < 0.1 * obj_hi {
                break;
            }
            solver.pop(1);
            solver.push();
            let mid = (obj_lo + obj_hi) / 2.0;
            solver.assert(
                &objective
                    .to_z3(&ctx, &f64_to_z3)
                    .le(&SymExpr::from(mid).to_z3(&ctx, &f64_to_z3)),
            );
            match solver.check() {
                z3::SatResult::Sat => {
                    println!("Solution found for these objective bounds.");
                }
                z3::SatResult::Unsat => {
                    println!("No solution for these objective bounds.");
                    obj_lo = mid;
                }
                z3::SatResult::Unknown => {
                    println!("Solver responded 'unknown' while optimizing the objective. Aborting optimization.");
                    println!(
                        "Reason for unknown: {}",
                        solver.get_reason_unknown().unwrap_or("none".to_owned())
                    );
                    break;
                }
            }
        }
        println!("Optimizing polynomial coefficients...");
        let optimized_solution = self.solve_lp_for_nonlinear_var_assignment(
            &best_bound,
            &best_nonlinear_var_assignment.unwrap(),
        )?;
        Ok(optimized_solution)
    }
}
